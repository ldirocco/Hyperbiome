import torch
from torch.utils.data import DataLoader
import os

from src.dataset import SketchDataset
from src.models import TransformerEmbedder
from src.trainer import train_model

from pytorch_metric_learning.losses import ProxyAnchorLoss

if __name__ == "__main__":

    print("Loading Seen Gallery...", flush=True)
    seen_gallery = SketchDataset('all_the_bacteria.sketch', 'metadata/assembly2species2directory.tsv', 'metadata/seen_gallery.csv')
    print("Done!", flush=True)

    input_dim = len(seen_gallery[0][0])
    num_classes = seen_gallery.n_classes

    print("Create dataloader...", flush=True)
    train_loader = DataLoader(seen_gallery, batch_size=32, num_workers=16, shuffle=True)
    print("Done!", flush=True)

    print("Model...", flush=True)
    model = TransformerEmbedder(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Done! (device: {device})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    proxy_loss_fn = ProxyAnchorLoss(num_classes, 128, margin=0.1, alpha=32)

    # Crea cartella output se non esiste
    os.makedirs('output', exist_ok=True)

    num_epochs = 20
    print("Start training...", flush=True)
    for epoch in range(num_epochs):
        loss = train_model(model, train_loader, optimizer, proxy_loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss:.4f}", flush=True)

    # Salva il modello finale
    model_path = 'output/metric_model.pth'
    model.to("cpu")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in {model_path}", flush=True)
