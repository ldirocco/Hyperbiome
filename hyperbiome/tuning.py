import argparse
import math
import torch
import optuna
import pickle

from functools import partial
from torch.utils.data import DataLoader

from hyperbiome.loss import HierarchicalProxyAnchor
from hyperbiome.models import HypTransformerEmbedder
from hyperbiome.train import evaluate
from hyperbiome.trainer import train_multiproxy_model
from hyperbiome.dataset import BacteriaSketches


def objective(
    trial,
    train_sketch_file,
    train_metadata,
    valid_sketch_file,
    valid_metadata,
    batch_size=32,
    num_workers=8,
    lr=1e-4,
    num_epochs=20,
    device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Hyperparameter Search ===
    dim = trial.suggest_int("dim", 128, 512, step=128)
    hyp_c = trial.suggest_float("curvature", 1e-3, 5.0, log=True)
    clip_r = 1.0 / (2.0 * math.sqrt(hyp_c))


    # === Dataset Loading ===
    print("Loading datasets...", flush=True)
    seen_gallery = BacteriaSketches(train_sketch_file, train_metadata, True)
    seen_query = BacteriaSketches(valid_sketch_file, valid_metadata, True)

    train_loader = DataLoader(
        seen_gallery, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valid_loader = DataLoader(
        seen_query, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    input_dim = len(seen_gallery[0][0])
    n_genera = seen_gallery.n_genera()
    n_species = seen_gallery.n_species()

    # === Model ===
    model = HypTransformerEmbedder(
        input_dim=input_dim, c=hyp_c, clip_r=clip_r, dim=dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === Loss Function ===
    proxy_loss_fn = HierarchicalProxyAnchor(
        n_genus=n_genera,
        n_species=n_species,
        sz_embed=dim,
        metadata_path=train_metadata,
        c=hyp_c,
        clip_r=clip_r,
        alpha=32,
    )

    # === Training Loop ===
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_multiproxy_model(
            model, train_loader, optimizer, proxy_loss_fn, device
        )

        val_loss = evaluate(model, valid_loader, proxy_loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}", flush=True)

        # Report progress to Optuna
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


# ===================== MAIN FUNCTION ===================== #
def run_tuning(train_sketch_file ,
                train_metadata,
                valid_sketch_file,
                valid_metadata
):

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
        train_sketch_file=train_sketch_file,
        train_metadata=train_metadata,
        valid_sketch_file=valid_sketch_file,
        valid_metadata=valid_metadata,
        batch_size=32,
        num_workers=8,
        lr=1e-4,
        num_epochs=20,
    )

    # Ottimizzazione
    study.optimize(objective_with_data, n_trials=100, gc_after_trial=True)
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
    parser.add_argument("--valid_sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--valid_metadata", type=str, required=True, help="Path al file dei metadata")



    args = parser.parse_args()

    run_tuning(train_sketch_file=args.train_sketch_file,
               train_metadata=args.train_metadata,
               valid_sketch_file=args.valid_sketch_file,
               valid_metadata=args.valid_metadata)