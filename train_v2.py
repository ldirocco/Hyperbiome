from prompt_toolkit.renderer import print_formatted_text
from torch.utils.data import DataLoader
import os
import argparse

from src.dataset import BacteriaSketches
from src.models import *
from src.loss import *
from src.trainer import train_model, train_multiproxy_model

from pytorch_metric_learning.losses import ProxyAnchorLoss

def main(args):
    print("Loading Seen Gallery...", flush=True)
    seen_gallery=BacteriaSketches(args.sketch_file,args.metadata,args.multi_proxy)
    print("Done!", flush=True)

    n_genera=seen_gallery.n_genera()
    n_species=seen_gallery.n_species()

    print(f"# Genera: {n_genera}", flush=True)
    print(f"# Species: {n_species}", flush=True)
    print(f"# Assemblies: {len(seen_gallery)}",flush=True)

    print("Create dataloader...", flush=True)
    train_loader = DataLoader(seen_gallery, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("Done!", flush=True)

    input_dim = len(seen_gallery[0][0])

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
        proxy_loss_fn = HypProxyAnchor(n_species, sz_embed=128, c=args.hyp_c, mrg=0.1, alpha=32)

    elif args.multi_proxy:
        print("Using Multi Proxy", flush=True)
        proxy_loss_fn=HypMultiProxyAnchor(n_genera,n_species, sz_embed=128, c=args.hyp_x, mrg=0.1, alpha=32)
    else:
        print("Using standard ProxyAnchorLoss", flush=True)
        proxy_loss_fn = ProxyAnchorLoss(n_species, 128, margin=0.1, alpha=32)


    print("Start training...", flush=True)
    if args.multi_proxy:
        for epoch in range(args.num_epochs):
            loss = train_multiproxy_model(model, train_loader, optimizer, proxy_loss_fn, device)
            print(f"Epoch {epoch + 1}/{args.num_epochs}: Loss = {loss:.4f}", flush=True)
    else:
        for epoch in range(args.num_epochs):
            loss = train_model(model, train_loader, optimizer, proxy_loss_fn, device)
            print(f"Epoch {epoch + 1}/{args.num_epochs}: Loss = {loss:.4f}", flush=True)


    os.makedirs(args.output_dir, exist_ok=True)

    if args.hyp:
        model_path = os.path.join(args.output_dir, 'hyp_metric_model.pth')
    elif args.multi_proxy:
        model_path = os.path.join(args.output_dir, 'multi_proxy_model.pth')
    else:
        model_path = os.path.join(args.output_dir, 'metric_model.pth')

    model.to("cpu")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in {model_path}", flush=True)

    if args.hyp:
        proxies_path = os.path.join(args.output_dir, 'hyp_proxies.pth')
        proxies = proxy_loss_fn.get_proxies()  # proxies iperbolici
        torch.save(proxies, proxies_path)
    elif args.multi_proxy:
        species_proxies_path = os.path.join(args.output_dir, 'species_proxies.pth')
        species_proxies = proxy_loss_fn.get_species_proxies()
        torch.save(species_proxies, species_proxies_path)
        genera_proxies_path = os.path.join(args.output_dir, 'genera_proxies.pth')
        genera_proxies = proxy_loss_fn.get_genera_proxies()
        torch.save(genera_proxies, genera_proxies_path)
    else:
        proxies_path = os.path.join(args.output_dir, 'eucl_proxies.pth')
        proxies = proxy_loss_fn.proxies.detach().cpu()  # proxies standard Euclidei
        torch.save(proxies, proxies_path)
    print(f"Done!!!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Embedder on Sketch Dataset")
    parser.add_argument("--sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--metadata", type=str, required=True, help="Path al file dei metadata")
    parser.add_argument("--output_dir", type=str, default="output_dir", help="Directory dove salvare il modello")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per il DataLoader")
    parser.add_argument("--num_workers", type=int, default=16, help="Numero di worker per il DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Numero di epoche per l'allenamento")
    parser.add_argument("--hyp", action='store_true', help="Se presente, usa HypTransformerEmbedder; altrimenti TransformerEmbedder")
    parser.add_argument("--hyp_c", type=float,default=0.1, help="Curvatura")
    parser.add_argument("--multi_proxy", action='store_true', help="Se presente, usa MultiProxy loss")




    args = parser.parse_args()
    main(args)