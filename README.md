
# 🧬 Hyperbiome

Hyperbiome is a **hyperbolic metric learning framework** for bacterial genome classification and comparison.  
It addresses the challenge of rapidly growing microbial databases by providing compact yet informative embeddings that reconstruct bacterial taxonomy and enable fast classification of new genomes.

---

## 🔬 Abstract

The bacterial kingdom remains largely unexplored, with new strains continuously being discovered.  
The increasing size of bacterial databases requires succinct but expressive representations that allow efficient classification and comparison of genomes.  

To meet this need, we propose **Hyperbiome**, a metric learning framework that leverages the geometry of the hyperbolic disk to:
- reconstruct bacterial taxonomy,
- learn a latent space where distances reflect biological similarities,
- incorporate taxonomic hierarchy in hyperbolic space,
- learn representative proxies at both species and genus levels.

Using species-level proxies, Hyperbiome builds a lightweight index that enables rapid classification of new assemblies without exhaustive query-vs-all scans.  

Experiments on **AllTheBacteria**, the largest bacterial database available, show that Hyperbiome effectively captures biological relationships and generalizes to unseen species. Moreover, our proxy-based index achieves high species-level classification accuracy while reducing memory and computational costs, outperforming state-of-the-art tools.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<user>/hyperbiome.git
cd hyperbiome
pip install -e .
````

To install the **HyperGen** backend (CPU/GPU):

```bash
hyperbiome hypergen install --type gpu --gpu ampere
```

---

## 🖥️ Command-Line Interface (CLI)

Hyperbiome provides a [Typer](https://typer.tiangolo.com/) powered CLI with multiple subcommands.

### 📥 Datasets

Download original or processed datasets:

```bash
hyperbiome datasets allthebacteria
hyperbiome datasets sketches
```

### ⚙️ Training

Train an embedding model:

```bash
hyperbiome train run \
    data/train.sketch data/train_metadata.tsv \
    data/valid.sketch data/valid_metadata.tsv \
    --output-dir outputs \
    --dim 128 \
    --hyp \
    --num-epochs 20 \
    --device gpu
```

### ✅ Validation

Validate a trained model:

```bash
hyperbiome valid run \
    outputs/model_checkpoint \
    --metadata-folder data/metadata \
    --sketches-folder data/sketches \
    --device gpu
```

### 🔍 Query

Run queries on new sequences:

```bash
hyperbiome query fasta --input new_sequences.fasta
```

### [HyperGen](https://github.com/wh-xu/Hyper-Gen)

Generate sketches from genomic data:

```bash
hyperbiome hypergen run \
    --type gpu \
    --data-folder data/fna \
    --output-file fna.sketch
```



---

## 📂 Project Structure

* `hyperbiome/train.py` – training logic for embedding models
* `hyperbiome/valid.py` – validation routines for trained models
* `hyperbiome/hypergen.py` – interface to HyperGen sketch generation
* `cli.py` – main CLI entry point (Typer-based)

---

## 📊 Results

TODO
---

## 📜 License

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for more details.
