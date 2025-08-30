from torch.utils.data import DataLoader
import os
import argparse

from src.dataset import SketchDataset
from src.models import *
from src.loss import *
from src.trainer import train_model

from pytorch_metric_learning.losses import ProxyAnchorLoss


def main(args):
    print("Loading Seen Gallery...", flush=True)
    #'all_the_bacteria.sketch', 'metadata/assembly2species2directory.tsv', 'metadata/seen_gallery.csv
    seen_gallery = SketchDataset(args.sketch_file, args.assembly_file, args.gallery_file)
    print("Done!", flush=True)

    input_dim = len(seen_gallery[0][0])
    num_classes = seen_gallery.n_classes

    print("Create dataloader...", flush=True)
    train_loader = DataLoader(seen_gallery, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("Done!", flush=True)

    print("Model...", flush=True)
    if args.hyp:
        print("Using HypTransformerEmbedder", flush=True)
        model = HypTransformerEmbedder(input_dim=input_dim)
    else:
        print("Using TransformerEmbedder", flush=True)
        model = TransformerEmbedder(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Done! (device: {device})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.hyp:
        print("Using HypProxyAnchor loss", flush=True)
        proxy_loss_fn = HypProxyAnchor(num_classes, sz_embed=128, c=args.hyp_c, mrg=0.1, alpha=32)
    else:
        print("Using standard ProxyAnchorLoss", flush=True)
        proxy_loss_fn = ProxyAnchorLoss(num_classes, 128, margin=0.1, alpha=32)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Start training...", flush=True)
    for epoch in range(args.num_epochs):
        loss = train_model(model, train_loader, optimizer, proxy_loss_fn, device)
        print(f"Epoch {epoch + 1}/{args.num_epochs}: Loss = {loss:.4f}", flush=True)

    if args.hyp:
        model_path = os.path.join(args.output_dir, 'hyp_metric_model.pth')
    else:
        model_path = os.path.join(args.output_dir, 'metric_model.pth')

    model.to("cpu")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in {model_path}", flush=True)

    proxies_path = os.path.join(args.output_dir, 'trained_proxies.pth')

    if args.hyp:
        proxies = proxy_loss_fn.get_proxies()  # proxies iperbolici
    else:
        proxies = proxy_loss_fn.proxies.detach().cpu()  # proxies standard Euclidei

    torch.save(proxies, proxies_path)
    print(f"Proxies allenati salvati in {proxies_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Embedder on Sketch Dataset")
    parser.add_argument("--sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--assembly_file", type=str, required=True, help="Path al file assembly2species2directory.tsv")
    parser.add_argument("--gallery_file", type=str, required=True, help="Path al file seen_gallery.csv")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory dove salvare il modello")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per il DataLoader")
    parser.add_argument("--num_workers", type=int, default=16, help="Numero di worker per il DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Numero di epoche per l'allenamento")
    parser.add_argument("--hyp", action='store_true', help="Se presente, usa HypTransformerEmbedder; altrimenti TransformerEmbedder")
    parser.add_argument("--hyp_c", type=float,default=0.1, help="Curvatura")



    args = parser.parse_args()
    main(args)
