import typer

from hyperbiome.train import run_train
from hyperbiome.tuning import run_tuning

app = typer.Typer(help="""
                        🦠 Hyperbiome
                        =================
                        
                        🌌 Explore the bacterial kingdom in HYPERBOLIC space!
                        
                        Commands available:
                         🏋️ train      Train the model to learn a metric space for bacterial strains
                         ⚡  tuning     Hyperparameter optimization with Optuna
                        
                        Use --help after any command for more details!
                        """)


#-----------------------------------------
#  Hyperbiome Commands
#-----------------------------------------

#  Parameters Tuning
@app.command("tuning", help="Hyperparameter⚡optimization⚡with⚡Optuna")
def hyperparameter_optimization(
      train_sketch_file: str = typer.Argument(..., help="📄 Training .sketch file"),
      train_metadata: str = typer.Argument(..., help="📄 Training metadata file"),
      valid_sketch_file: str = typer.Argument(..., help="📄 Validation .sketch file"),
      valid_metadata: str = typer.Argument(..., help="📄 Validation metadata file")
):
    typer.echo("⚡ Launching Optuna hyperparameter tuning... 🧪")

    run_tuning(
        train_sketch_file=train_sketch_file,
        train_metadata=train_metadata,
        valid_sketch_file=valid_sketch_file,
        valid_metadata=valid_metadata
    )

#  Training the model
@app.command("train", help="Train metric learning model")
def learn_metric_space(
        train_sketch_file: str = typer.Argument(..., help="Path to the training sketches file"),
        train_metadata: str = typer.Argument(..., help="Path to the training metadata file"),
        valid_sketch_file: str = typer.Argument(..., help="Path to the validation sketches file"),
        valid_metadata: str = typer.Argument(..., help="Path to the validation metadata file"),
        output_dir: str = typer.Option("outputs", "--output-dir", "-o", envvar="OUTPUT_DIR",
                                       help="Output directory"),
        dim: int = typer.Option(128, "--dim", "-d", help="Embedding dimension"),
        hyp: bool = typer.Option(False, "--hyp", help="Enable projection of bacteria into the Poincaré disk"),
        taxonomy_proxies: bool = typer.Option(False, "--taxonomy",
                                              help="Enable taxonomy proxies to model the Genus–Species hierarchical structure"),
        c: float = typer.Option(0.1, "--c", help="Curvature"),
        r: float = typer.Option(2.3, "--r", help="Clipping radius"), #Modifica con moderate strategy
        batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for DataLoader"),
        num_workers: int = typer.Option(16, "--num-workers", "-w", help="Number of workers for DataLoader"),
        lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
        num_epochs: int = typer.Option(20, "--num-epochs", help="Number of training epochs"),
        scheduler: str = typer.Option("plateau", "--scheduler", show_choices=True, help="LR scheduler type",
                                      metavar="[none|plateau|step]"),
        plateau_mode: str = typer.Option("min", "--plateau-mode", show_choices=True,
                                         help="ReduceLROnPlateau monitor mode", metavar="[min|max]"),
        factor: float = typer.Option(0.5, "--factor", help="ReduceLROnPlateau: LR reduction factor"),
        patience: int = typer.Option(3, "--patience",
                                     help="ReduceLROnPlateau: epochs without improvement before reducing LR"),
        threshold: float = typer.Option(1e-4, "--threshold", help="ReduceLROnPlateau: improvement threshold"),
        cooldown: int = typer.Option(0, "--cooldown", help="ReduceLROnPlateau: cooldown after reduction"),
        min_lr: float = typer.Option(1e-7, "--min-lr", help="ReduceLROnPlateau: minimum LR"),
        step_size: int = typer.Option(10, "--step-size", help="StepLR: epochs per step"),
        gamma: float = typer.Option(0.1, "--gamma", help="StepLR: decay factor"),
        early_stop_patience: int = typer.Option(10, "--early-stop-patience",
                                                help="Early stopping: epochs without improvement"),
        early_stop_min_delta: float = typer.Option(0.0, "--early-stop-min-delta",
                                                   help="Early stopping: minimum improvement to reset patience"),
        device: str = typer.Option("cpu", "--device", "-v", help="Device for training (cpu/gpu)"),
):
    typer.echo("🌌 Embedding bacteria strains...")

    run_train(
        train_sketch_file=train_sketch_file,
        train_metadata=train_metadata,
        valid_sketch_file=valid_sketch_file,
        valid_metadata=valid_metadata,
        output_dir=output_dir,
        dim=dim,
        hyp=hyp,
        taxonomy_proxies=taxonomy_proxies,
        c=c,
        r=r,  # Modifica con moderate strategy
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

if __name__ == "__main__":
    app()
