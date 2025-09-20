#!/usr/bin/env python3
import subprocess
import sys
import os
import typer

# str = typer.Option("cpu", "--type", "-t", help="Install type: cpu or gpu"),
#str = typer.Option("ada-lovelace", "--gpu", "-g", help="GPU type: ada-lovelace, ampere, volta, hopper")
def run_cmd(cmd: str):
    """Run a shell command and stop on error"""
    print(f"💻 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(result.returncode)

def install_hypergen(
    device = "cpu",
    gpu_type = "ampere"
):
    """
    Install Hyper-Gen with optional GPU support.
    """
    # ----------------------------
    # Check cargo
    # ----------------------------
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.DEVNULL)
        print("✅ Cargo found")
    except subprocess.CalledProcessError:
        print("🔧 Cargo not found. Installing Rust and Cargo...")
        run_cmd("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        # Add cargo to PATH for current session
        cargo_env = os.path.expanduser("~/.cargo/env")
        if os.path.exists(cargo_env):
            run_cmd(f"source {cargo_env}")

    # ----------------------------
    # Clone Hyper-Gen if needed
    # ----------------------------
    if not os.path.isdir("Hyper-Gen"):
        print("📥 Cloning Hyper-Gen repository...")
        run_cmd("git clone https://github.com/wh-xu/Hyper-Gen.git")

    os.chdir("Hyper-Gen")

    # ----------------------------
    # Run cargo install
    # ----------------------------
    if device == "cpu":
        print("🖥️ Installing CPU version...")
        run_cmd("cargo install --path .")
    elif device == "gpu":
        print(f"⚡ Installing GPU version for {gpu_type}...")
        gpu_map = {
            "ada-lovelace": "cuda-sketch-ada-lovelace",
            "ampere": "cuda-sketch-ampere",
            "volta": "cuda-sketch-volta",
            "hopper": "cuda-sketch-hopper"
        }
        if gpu_type not in gpu_map:
            print(f"❌ Unknown GPU type: {gpu_type}")
            sys.exit(1)
        feature = gpu_map[gpu_type]
        run_cmd(f"cargo install --features {feature} --path .")
    else:
        print(f"❌ Unknown install type: {device}")
        sys.exit(1)

    print("✅ Hyper-Gen installation completed!")


def run_hypergen(
    device = "cpu",
    data_folder = "data",
    output_file = "fna.sketch",
):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if device == "cpu":
        run_cmd(f"hyper-gen sketch -p {data_folder} -o {output_file}")
    elif device == "gpu":
        run_cmd(f"hyper-gen sketch -D gpu -p {data_folder} -o {output_file}")
    else:
        print(f"❌ Unknown device type: {device}")
        sys.exit(1)
