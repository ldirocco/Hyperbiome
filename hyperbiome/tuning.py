import argparse
import math
import torch
import optuna
import os
from tqdm import tqdm
import pickle
import numpy as np

from functools import partial

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from back.recall_k import recall_at_k
from hyperbiome.loss import HierarchicalProxyAnchor
from hyperbiome.modules import *
from hyperbiome.models import HypTransformerEmbedder
from hyperbiome.train import evaluate
from hyperbiome.trainer import train_multiproxy_model
from hyperbiome.dataset import BacteriaSketches_optuna

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)



def compute_recall_k(model, k, train_loader, valid_loader,c, device):
    model.eval()


    def compute_embedding(dataloader, model):
        emb = []
        labels = []
        with torch.no_grad():
            for x, y_species, y_genus in tqdm(dataloader, leave=False,disable=True):
                x, y_species, y_genus = x.to(device), y_species.to(device), y_genus.to(device)
                emb.append(model(x))
                labels.append(y_species)
        return torch.cat(emb, dim=0), torch.cat(labels, dim=0)

    emb_train, labels_train = compute_embedding(train_loader, model)
    emb_val, labels_val = compute_embedding(valid_loader, model)

    dists = pmath.dist_matrix(emb_val, emb_train, c)#torch.cdist(emb_val, emb_train)
    _, topk_indices = torch.topk(-dists, k, dim=1)
    correct = labels_train[topk_indices].eq(labels_val[:, None])

    return (correct.sum(1) / k).mean()


def objective(
    trial,
    train_set,
    val_set,
    k=11,
    num_workers=8,
    lr=1e-4,
    num_epochs=100,
    device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_id = trial.number % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)

    # === Hyperparameter Search ===
    depth = trial.suggest_int("depth", 4, 16, step=4)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    c = trial.suggest_float("curvature", 1e-3, 5.0, log=True)
    # === Setting Clipping Radius
    r = 1.0 / (2.0 * math.sqrt(c))


    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valid_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    input_dim = len(train_set[0][0])
    n_genera = train_set.dataset.n_genera
    n_species = train_set.dataset.n_species

    # === Model ===
    model = HypTransformerEmbedder(
        input_dim=input_dim, c=c, clip_r=r, depth=depth
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === Loss Function ===
    proxy_loss = HierarchicalProxyAnchor(
        n_genus=n_genera,
        n_species=n_species,
        sz_embed=128,
        c=c,
        clip_r=r,
        alpha=32,
    )

    # === Scheduler ===
    early_stop_patience = 10,

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=0.0001,
        cooldown=0,
        min_lr=0.0000001
    )

    # === Training Loop ===

    best_val = float("inf")
    epochs_no_improve = 0

    early_stop_min_delta = 0.0

    for epoch in range(num_epochs):

        train_multiproxy_model(
            model, train_loader, optimizer, proxy_loss, device
        )

        val_loss = evaluate(model, valid_loader, proxy_loss, device)

        if scheduler == "plateau":
            scheduler.step(val_loss)

        # Early stopping + save best
        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping dopo {epoch + 1} epoche (best val_loss={best_val:.4f})", flush=True)
                break

        recall_at_k= compute_recall_k(model, k, train_set, val_set,c,device)

        trial.report(recall_at_k, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return recall_at_k




# ===================== MAIN FUNCTION ===================== #
def run_tuning(train_sketch_file ,
               train_metadata,
               s,
               device=None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")
    print("Loading sketch ...")
    #Take 200 samples for each class
    seen_gallery = BacteriaSketches_optuna(
        train_sketch_file,
        train_metadata,
        s=s,
        return_genus=True
    )

    print("...done")

    targets = np.array(seen_gallery.labels_df.Species_ID) # estrai label relative al subset
    indices = np.arange(len(seen_gallery))

    train_idx, val_idx = [], []
    for c in np.unique(targets):
        class_indices = indices[targets == c]
        train_c, val_c = train_test_split(class_indices, test_size= int(s*0.2), random_state=42, stratify=None)
        train_idx.extend(train_c)
        val_idx.extend(val_c)

    train_set = Subset(seen_gallery, train_idx)
    val_set = Subset(seen_gallery, val_idx)


    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")

    # Creazione dello studio Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,  # Risorse minime per iniziare il pruning (es. epoche)
            reduction_factor=3,  # Fattore di riduzione dei trial in ciascun step
            min_early_stopping_rate=0  # Numero minimo di step prima del pruning
        ),
    )

    # Funzione objective "parzialmente applicata"
    objective_with_data = partial(
        objective,
        train_set=train_set,
        val_set=val_set,
        num_workers=8,
        lr=1e-4,
        num_epochs=100,
        k=11,
        device=device
    )

    # Ottimizzazione
    study.optimize(objective_with_data, n_trials=100, n_jobs=8, gc_after_trial=True, show_progress_bar=True)
    # Path al file pickle
    study_path = "optuna_results/study.pkl"

    with open(study_path, "wb") as f:
        pickle.dump(study, f)


    # with open(study_path, "rb") as f:
    #     study = pickle.load(f)
    # best_params = study.best_params
    # print("Best Hyperparameters:", best_params)
    # best_value = study.best_value
    # print("Best Objective Value:", best_value)
    # trials = study.trials
    # print("Number of trials:", len(trials))
    # for t in trials:
    #     print(f"Trial {t.number} | Value: {t.value} | Params: {t.params} | State: {t.state}")
    # from optuna.visualization import (
    #     plot_optimization_history,
    #     plot_parallel_coordinate,
    #     plot_slice,
    #     plot_param_importances
    # )
    # plot_optimization_history(study).show()
    # plot_parallel_coordinate(study).show()
    # plot_slice(study).show()
    # plot_param_importances(study).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tuning hyperparameters for hyperbiome"
    )

    parser.add_argument("--train_sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--train_metadata", type=str, required=True, help="Path al file dei metadata")
    parser.add_argument("--s", type=int, required=True, help="Path al file .sketch")



    args = parser.parse_args()

    run_tuning(train_sketch_file=args.train_sketch_file,
               train_metadata=args.train_metadata,
               s=int(args.s))