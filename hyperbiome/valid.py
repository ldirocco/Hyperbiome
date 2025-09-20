import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from hyperbiome.dataset import *
from hyperbiome.models import *
from hyperbiome.modules import HypProjector
import hyperbiome.poincare_math as pmath

import os
import argparse


def build_new_proxies(model, dataset, device, c=0.01, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings_by_class = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Embedding unseen_gallery"):
            X = X.float().to(device)
            y = y.to(device)
            emb = model(X)
            for e, label in zip(emb, y):
                embeddings_by_class[int(label.item())].append(e.cpu())

    if len(embeddings_by_class) == 0:
        return torch.empty(0), []

    new_labels = sorted(embeddings_by_class.keys())
    new_proxies_list = []
    for cls in new_labels:
        vecs = embeddings_by_class[cls]            # [D]
        mat = torch.stack(vecs, dim=0)            # [N_cls, D]
        mean_p = pmath.poincare_mean(mat, dim=0, c=c)   # [D]
        mean_p = pmath.project(mean_p, c=c)
        new_proxies_list.append(mean_p)

    new_proxies = torch.stack(new_proxies_list, dim=0).to(device).float()  # [num_new, D] su device
    return new_proxies, new_labels
    
def run_valid(model_folder, c, r, metadata_folder, sketches_folder, device):
    print(device, flush=True)

    # model
    model_path = os.path.join(model_folder, "multi_proxy_model.pth")
    model = HypTransformerEmbedder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # proxies
    proxies_path = os.path.join(model_folder, "species_proxies.pth")
    proxies_tan = torch.load(proxies_path).to(device).float()

    projector = HypProjector(c=c, riemannian=True, clip_r=r)
    proxies = projector(proxies_tan)

    print("Loading Unseen Gallery", flush=True)
    unseen_gallery = BacteriaSketches(
        os.path.join(sketches_folder, "unseen_gallery.sketch"),
        os.path.join(metadata_folder, "labeled_unseen_gallery.csv"),
        return_genus=False
    )

    print("Building new proxies", flush=True)
    new_proxies, new_labels = build_new_proxies(model, unseen_gallery, device)

    all_proxies = torch.cat([proxies, new_proxies], dim=0)


    print("Loading Unseen Query", flush=True)
    unseen_query = BacteriaSketches(
        os.path.join(sketches_folder, "unseen_query.sketch"),
        os.path.join(metadata_folder, "labeled_unseen_query.csv"),
        return_genus=False
    )
    loader = DataLoader(unseen_query, batch_size=128, shuffle=False)

    total = len(unseen_query)
    correct = 0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Classifying unseen queries"):
            X = X.float().to(device)
            y = y.to(device)

            embeddings = model(X)  # [B, D]
            dists = pmath.dist_matrix(embeddings, all_proxies, c=0.01)  # [B, num_total]
            preds = torch.argmin(dists, dim=1)
            correct += (preds == y).sum().item()

    accuracy = correct / total
    print(f"Accuracy on unseen queries: {accuracy*100:.2f}%", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegui la validazione del modello con i dati forniti"
    )

    parser.add_argument(
        "--device", type=str, default="", help="device"
    )
    parser.add_argument(
        "--model_folder", type=str, required=True,
        help="Cartella del modello addestrato"
    )
    parser.add_argument(
        "--c", type=float, required=True,
        help="Parametro c (intero)"
    )
    parser.add_argument(
        "--r", type=float, required=True,
        help="Parametro r (intero)"
    )
    parser.add_argument(
        "--metadata_folder", type=str, required=True,
        help="Cartella contenente i file di metadata"
    )
    parser.add_argument(
        "--sketches_folder", type=str, required=True,
        help="Cartella contenente gli sketch"
    )

    args = parser.parse_args()

    if args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    run_valid(
        model_folder=args.model_folder,
        c=args.c,
        r=args.r,
        metadata_folder=args.metadata_folder,
        sketches_folder=args.sketches_folder,
        device=device,
    )

## ESEMPIO
# python valid.py --model_folder "/home/terastiles/Hyperbiome/outputs/dim_128__c_0_01__r_5_0" --c 0.01 --r 5.0  --metadata_folder "/home/terastiles/Hyperbiome/data/metadata" --sketches_folder "/home/terastiles/Hyperbiome/data/sketches"

