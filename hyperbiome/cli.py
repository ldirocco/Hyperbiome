import typer
from hyperbiome.tuning import run_tuning

app = typer.Typer(help="""
                        🦠 Hyperbiome CLI
                        =================
                        
                        🌌 Explore the bacterial kingdom in HYPERBOLIC space!
                        
                        Commands available:
                        
                          ⚡ tuning     Hyperparameter optimization with Optuna
                        
                        Use --help after any command for more details!
                        """)


#-----------------------------------------
#  Hyperbiome Commands
#-----------------------------------------

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

if __name__ == "__main__":
    app()
