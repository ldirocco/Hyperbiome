import typer
from rich.progress import track
from hyperbiome.train import run_train
from hyperbiome.valid import run_valid
from hyperbiome.hypergen import install_hypergen, run_hypergen

app = typer.Typer(help="Hyperbiome - CLI for training and validating embedding models")

# Sub-apps for training and validation
hypergen_app = typer.Typer(help="Commands for Hyper-Gen")
datasets_app = typer.Typer(help="Commands for download datasets")
train_app = typer.Typer(help="Commands for training")
valid_app = typer.Typer(help="Commands for validation")
query_app = typer.Typer(help="Commands for running query")

# Global callback
@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode")):
    if verbose:
        typer.echo("🔍 Verbose mode activated")

# -----------------------------------------
# TRAIN COMMAND
# -----------------------------------------
@train_app.command("run", help="🚀 Train Autoencoder/Metric Learning")
def train(
    train_sketch_file: str = typer.Argument(..., help="📄 Training .sketch file"),
    train_metadata: str = typer.Argument(..., help="📄 Training metadata file"),
    valid_sketch_file: str = typer.Argument(..., help="📄 Validation .sketch file"),
    valid_metadata: str = typer.Argument(..., help="📄 Validation metadata file"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", envvar="OUTPUT_DIR", help="📂 Directory to save model"),
    dim: int = typer.Option(128, "--dim", "-d", help="Embedding dimension"),
    multi_proxy: bool = typer.Option(False, "--multi-proxy", help="Use MultiProxy loss"),
    hyp: bool = typer.Option(False, "--hyp", help="Use HypTransformerEmbedder"),
    c: float = typer.Option(0.1, "--c", help="Curvature"),
    r: float = typer.Option(2.3, "--r", help="Clipping radius"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for DataLoader"),
    num_workers: int = typer.Option(16, "--num-workers", "-w", help="Number of workers for DataLoader"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    num_epochs: int = typer.Option(20, "--num-epochs", help="Number of training epochs"),
    scheduler: str = typer.Option("plateau", "--scheduler", show_choices=True, help="LR scheduler type", metavar="[none|plateau|step]"),
    plateau_mode: str = typer.Option("min", "--plateau-mode", show_choices=True, help="ReduceLROnPlateau monitor mode", metavar="[min|max]"),
    factor: float = typer.Option(0.5, "--factor", help="ReduceLROnPlateau: LR reduction factor"),
    patience: int = typer.Option(3, "--patience", help="ReduceLROnPlateau: epochs without improvement before reducing LR"),
    threshold: float = typer.Option(1e-4, "--threshold", help="ReduceLROnPlateau: improvement threshold"),
    cooldown: int = typer.Option(0, "--cooldown", help="ReduceLROnPlateau: cooldown after reduction"),
    min_lr: float = typer.Option(1e-7, "--min-lr", help="ReduceLROnPlateau: minimum LR"),
    step_size: int = typer.Option(10, "--step-size", help="StepLR: epochs per step"),
    gamma: float = typer.Option(0.1, "--gamma", help="StepLR: decay factor"),
    early_stop_patience: int = typer.Option(10, "--early-stop-patience", help="Early stopping: epochs without improvement"),
    early_stop_min_delta: float = typer.Option(0.0, "--early-stop-min-delta", help="Early stopping: minimum improvement to reset patience"),
    device: str = typer.Option("cpu", "--device", "-v", help="Device for training (cpu/gpu)"),
):
    typer.echo(f"🏋️ Starting training for {num_epochs} epochs...")
    # Simulated training progress with Rich
    for epoch in track(range(num_epochs), description="Training..."):
        pass  # replace with your run_train logic
    # Actual training call
    run_train(
        train_sketch_file=train_sketch_file,
        train_metadata=train_metadata,
        valid_sketch_file=valid_sketch_file,
        valid_metadata=valid_metadata,
        output_dir=output_dir,
        dim=dim,
        multi_proxy=multi_proxy,
        hyp=hyp,
        hyp_c=c,
        clip_r=r,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        num_epochs=num_epochs,
        scheduler=scheduler,
        plateau_mode=plateau_mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        cooldown=cooldown,
        min_lr=min_lr,
        step_size=step_size,
        gamma=gamma,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        device=device
    )

# -----------------------------------------
# VALID COMMAND
# -----------------------------------------
@valid_app.command("run", help="✅ Validate a trained model")
def valid(
    model_folder: str = typer.Argument(..., help="📂 Folder containing the trained model"),
    metadata_folder: str = typer.Option(..., help="📄 Folder with metadata"),
    sketches_folder: str = typer.Option(..., help="✏️ Folder with sketches"),
    c: float = typer.Option(0.1, "--c", help="Curvature"),
    r: float = typer.Option(2.3, "--r", help="Clipping radius"),
    device: str = typer.Option("cpu", "--device", help="Device (cpu/gpu)"),
):
    typer.echo(f"🧪 Starting validation for model in {model_folder}")
    run_valid(
        model_folder=model_folder,
        c=c,
        r=r,
        metadata_folder=metadata_folder,
        sketches_folder=sketches_folder,
        device=device,
    )


@hypergen_app.command("install", help="Install HyperGen")
def install_hypergen_app(
        device: str = typer.Option("cpu", "--type", "-t", help="Install type: cpu or gpu"),
        gpu_type: str = typer.Option("ampere", "--gpu", "-g", help="GPU type: ada-lovelace, ampere, volta, hopper")
):
    install_hypergen(device =device, gpu_type = gpu_type)

@hypergen_app.command("run", help="Run HyperGen")
def run_hypergen_app(
        device: str = typer.Option("cpu", "--type", "-t", help="Run type: cpu or gpu"),
        data_folder: str = typer.Option("data", "--device", help="Folder with data"),
        output_file: str = typer.Option("fna.sketch", "--device", help="Name of output file"),
):
    run_hypergen(device=device, data_folder=data_folder, output_file=output_file)

@datasets_app.command("allthebacteria", help="Download AllTheBacteria Original Dataset")
def allthebacteria():
    pass

@datasets_app.command("sketches", help="Download AllTheBacteria processed Dataset with HyperGen")
def allthebacteria_processed():
    pass


@query_app.command("fasta", help="Run Fasta Query")
def fasta_query():
    pass

# -----------------------------------------
# Attach subcommands to the main app
# -----------------------------------------
app.add_typer(datasets_app, name="datasets")
app.add_typer(hypergen_app, name="hypergen")
app.add_typer(train_app, name="train")
app.add_typer(valid_app, name="valid")
app.add_typer(query_app, name="query")


# Entry point
if __name__ == "__main__":
    app()
