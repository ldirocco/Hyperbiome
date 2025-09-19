import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader

from hyperbiome.dataset import BacteriaSketches
from hyperbiome.models import *
from hyperbiome.loss import *
from hyperbiome.trainer import train_model, train_multiproxy_model

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
)

from pytorch_metric_learning.losses import ProxyAnchorLoss



@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y_species, y_genus in loader:
        x, y_species, y_genus = x.to(device), y_species.to(device), y_genus.to(device)
        emb = model(x)
        loss = loss_fn(emb, y_species, y_genus)
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / len(loader)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_train(train_sketch_file,
          train_metadata,
          valid_sketch_file, 
          valid_metadata, 
          output_dir="outputs", 
          dim=128, 
          multi_proxy=True, 
          hyp=False, 
          hyp_c=0.1, 
          clip_r=2.3, 
          batch_size=32, 
          num_workers=16, 
          lr=0.0001, 
          num_epochs=20, 
          scheduler="plateau", 
          plateau_mode="min", 
          factor=0.5, 
          patience=3, 
          threshold=0.0001, 
          cooldown=0, 
          min_lr=0.0000001, 
          step_size=10, 
          gamma=0.1, 
          early_stop_patience=10, 
          early_stop_min_delta=0.0,
          device="cpu"):
    print("Loading Seen Gallery...", flush=True)
    seen_gallery = BacteriaSketches(train_sketch_file, train_metadata, multi_proxy)
    print("Done!", flush=True)

    print("Loading Seen Query...", flush=True)
    seen_query = BacteriaSketches(valid_sketch_file, valid_metadata, multi_proxy)
    print("Done!", flush=True)

    n_genera = seen_gallery.n_genera()
    n_species = seen_gallery.n_species()

    if multi_proxy:
        print(f"# Genera: {n_genera}", flush=True)
        print(f"# Species: {n_species}", flush=True)
        print(f"# Assemblies: {len(seen_gallery)}", flush=True)

    print("Create train dataloader...", flush=True)
    train_loader = DataLoader(
        seen_gallery,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    print("Done!", flush=True)

    print("Create valid dataloader...", flush=True)
    valid_loader = DataLoader(
        seen_query,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    print("Done!", flush=True)

    input_dim = len(seen_gallery[0][0])

    print("Model...", flush=True)
    if hyp or multi_proxy:
        print("Using HypTransformerEmbedder", flush=True)
        model = HypTransformerEmbedder(
            input_dim=input_dim, c=hyp_c, clip_r=clip_r, dim=dim
        )
    else:
        print("Using TransformerEmbedder", flush=True)
        model = TransformerEmbedder(input_dim=input_dim, dim=dim)


    model.to(device)
    print(f"Done! (device: {device})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    if multi_proxy:
        print("Using Multi Proxy", flush=True)
        proxy_loss_fn = HypMultiProxyAnchor(
            n_genera,
            n_species,
            sz_embed=dim,
            metadata_path=train_metadata,
            c=hyp_c,
            clip_r=clip_r,
            mrg=0.1,
            alpha=32,
        )
    elif hyp:
        print("Using HypProxyAnchor loss", flush=True)
        proxy_loss_fn = HypProxyAnchor(
            n_species,
            sz_embed=dim,
            c=hyp_c,
            clip_r=clip_r,
            mrg=0.1,
            alpha=32,
        )
    else:
        print("Using standard ProxyAnchorLoss", flush=True)
        proxy_loss_fn = ProxyAnchorLoss(n_species, dim, margin=0.1, alpha=32)

    # Scheduler
    if scheduler == "none":
        scheduler =  None
    elif scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=plateau_mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
        )
    elif scheduler == "step":
        scheduler =  StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    else:
        raise ValueError(f"Scheduler non supportato: {scheduler}")

    # Preparazione logging CSV
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    print("Start training...", flush=True)

    best_val = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    if multi_proxy:
        for epoch in range(num_epochs):
            # Training
            train_loss = train_multiproxy_model(
                model, train_loader, optimizer, proxy_loss_fn, device
            )
            # Validation
            val_loss = evaluate(model, valid_loader, proxy_loss_fn, device)

            # Logging
            print(
                f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr(optimizer):.2e}",
                flush=True,
            )
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr(optimizer):.6e}"])

            # Scheduler step (usa val_loss per plateau)
            if scheduler is not None:
                if scheduler == "plateau":
                    scheduler.step(val_loss)
                elif scheduler in ("step",):
                    scheduler.step()

            # Early stopping + save best
            if val_loss < best_val - early_stop_min_delta:
                best_val = val_loss
                epochs_no_improve = 0
                # Salva best model
                torch.save(model.state_dict(), best_model_path)
                # Salva anche i proxies migliori
                species_proxies_path = os.path.join(output_dir, "best_species_proxies.pth")
                genera_proxies_path = os.path.join(output_dir, "best_genera_proxies.pth")
                species_proxies = proxy_loss_fn.get_species_proxies()
                genera_proxies = proxy_loss_fn.get_genera_proxies()
                torch.save(species_proxies, species_proxies_path)
                torch.save(genera_proxies, genera_proxies_path)
                print(f"Best model aggiornato (val_loss={best_val:.4f})", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping dopo {epoch + 1} epoche (best val_loss={best_val:.4f})", flush=True)
                    break
    else:
        for epoch in range(num_epochs):
            # Training
            train_loss = train_model(
                model, train_loader, optimizer, proxy_loss_fn, device
            )
            # Validation
            val_loss = evaluate(model, valid_loader, proxy_loss_fn, device)

            # Logging
            print(
                f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr(optimizer):.2e}",
                flush=True,
            )
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr(optimizer):.6e}"])

            # Scheduler step (usa val_loss per plateau)
            if scheduler is not None:
                if scheduler == "plateau":
                    scheduler.step(val_loss)
                elif scheduler in ("step",):
                    scheduler.step()

            # Early stopping + save best
            if val_loss < best_val - early_stop_min_delta:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                # Salva proxies migliori
                best_proxies_path = os.path.join(output_dir, "best_proxies.pth")
                proxies = proxy_loss_fn.proxies.detach().cpu()
                torch.save(proxies, best_proxies_path)
                print(f"Best model aggiornato (val_loss={best_val:.4f})", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping dopo {epoch + 1} epoche (best val_loss={best_val:.4f})", flush=True)
                    break

    # Salvataggi finali (ultimo stato addestrato)
    if hyp:
        model_path = os.path.join(output_dir, "hyp_metric_model.pth")
    elif multi_proxy:
        model_path = os.path.join(output_dir, "multi_proxy_model.pth")
    else:
        model_path = os.path.join(output_dir, "metric_model.pth")

    model.to("cpu")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in {model_path}", flush=True)

    if hyp:
        proxies_path = os.path.join(output_dir, "hyp_proxies.pth")
        proxies = proxy_loss_fn.get_proxies()
        torch.save(proxies, proxies_path)
    elif multi_proxy:
        species_proxies_path = os.path.join(output_dir, "species_proxies.pth")
        species_proxies = proxy_loss_fn.get_species_proxies()
        torch.save(species_proxies, species_proxies_path)

        genera_proxies_path = os.path.join(output_dir, "genera_proxies.pth")
        genera_proxies = proxy_loss_fn.get_genera_proxies()
        torch.save(genera_proxies, genera_proxies_path)
    else:
        proxies_path = os.path.join(output_dir, "eucl_proxies.pth")
        proxies = proxy_loss_fn.proxies.detach().cpu()
        torch.save(proxies, proxies_path)

    print("Done!!!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Transformer Embedder on Sketch Dataset"
    )

    # I/O
    parser.add_argument("--train_sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--train_metadata", type=str, required=True, help="Path al file dei metadata")
    parser.add_argument("--valid_sketch_file", type=str, required=True, help="Path al file .sketch")
    parser.add_argument("--valid_metadata", type=str, required=True, help="Path al file dei metadata")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory dove salvare il modello")

    # Model setup
    parser.add_argument("--dim", type=int, default=128, help="Dimensione spazio embedding")
    parser.add_argument("--multi_proxy", action="store_true", help="Se presente, usa MultiProxy loss")
    parser.add_argument("--hyp", action="store_true",
                        help="Se presente, usa HypTransformerEmbedder; altrimenti TransformerEmbedder")
    parser.add_argument("--hyp_c", type=float, default=0.1, help="Curvatura")
    parser.add_argument("--clip_r", type=float, default=2.3, help="Clipping radius")

    # Train setup
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per il DataLoader")
    parser.add_argument("--num_workers", type=int, default=16, help="Numero di worker per il DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Numero di epoche per l'allenamento")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau", "step"],
        help="Tipo di LR scheduler",
    )

    ## ReduceLROnPlateau
    parser.add_argument("--plateau_mode", type=str, default="min", choices=["min", "max"], help="Plateau: modalità di monitoraggio")
    parser.add_argument("--factor", type=float, default=0.5, help="Plateau: fattore di riduzione LR")
    parser.add_argument("--patience", type=int, default=3, help="Plateau: epoche senza miglioramento prima della riduzione")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Plateau: soglia di miglioramento")
    parser.add_argument("--cooldown", type=int, default=0, help="Plateau: cooldown dopo riduzione")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Plateau: LR minimo")

    ## StepLR
    parser.add_argument("--step_size", type=int, default=10, help="StepLR: epoche per step")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR: fattore di decay")

    ## Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Epoche senza miglioramento per early stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Miglioramento minimo sulla val_loss per reset della pazienza")

    parser.add_argument(
        "--device", type=str, default="", help="device"
    )
    args = parser.parse_args()

    if args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    run_train(
        train_sketch_file=args.train_sketch_file,
        train_metadata=args.train_metadata,
        valid_sketch_file=args.valid_sketch_file,
        valid_metadata=args.valid_metadata,
        output_dir=args.output_dir,
        dim=args.dim,
        multi_proxy=args.multi_proxy,
        hyp=args.hyp,
        hyp_c=args.hyp_c,
        clip_r=args.clip_r,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        num_epochs=args.num_epochs,
        scheduler=args.scheduler,
        plateau_mode=args.plateau_mode,
        factor=args.factor,
        patience=args.patience,
        threshold=args.threshold,
        cooldown=args.cooldown,
        min_lr=args.min_lr,
        step_size=args.step_size,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        device=device
)
