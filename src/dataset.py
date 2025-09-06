import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import read_file_sketch, decompress_hd_sketch

class BacteriaSketches(Dataset):
    def __init__(self, sketch_path, labels_path, return_genus=True):
        with open(sketch_path, 'rb') as f:
            self.sketches = read_file_sketch(f)

        self.sample = []
        for sketch in self.sketches:
            path = Path(sketch['file_str'])
            file_name = path.name
            self.sample.append(file_name[:-3])
            labels_df=pd.read_csv(labels_path, sep='\t')
            self.genus_labels=dict(zip(labels_df["Sample"], labels_df["Genus_ID"]))
            self.species_labels=dict(zip(labels_df["Sample"], labels_df["Species_ID"]))
            self.return_genus=return_genus

    def n_genera(self):
        return len(np.unique(np.array(list(self.genus_labels.values()))))

    def n_species(self):
        return len(np.unique(np.array(list(self.species_labels.values()))))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        sketch = self.sketches[idx]
        path = Path(sketch['file_str'])
        file_name = path.name
        sample = file_name[:-3]

        hv = decompress_hd_sketch(sketch)
        hv = torch.tensor(hv, dtype=torch.float32) / 255.0

        species_id = torch.tensor(self.species_labels[sample], dtype=torch.long)

        if self.return_genus:
            genus_id = torch.tensor(self.genus_labels[sample], dtype=torch.long)
            return hv, species_id, genus_id
        else:
            return hv, species_id


class SketchDataset(Dataset):
    def __init__(self, sketch_path, metadata_path, filter_path=None):
        """
        path: sketch file path
        """
        with open(sketch_path, 'rb') as f:
            self.sketches = read_file_sketch(f)

        if filter is not None:
            keep_these_assemblies=set(pd.read_csv(filter_path, sep='\t')["Sample"])
            self.sketches = [s for s in self.sketches if Path(s["file_str"]).name[:-3] in keep_these_assemblies]

        self.sample = []
        for sketch in self.sketches:
            path = Path(sketch['file_str'])
            file_name = path.name
            self.sample.append(file_name[:-3])

        self.metadata = pd.read_csv(metadata_path, sep='\t')
        self.metadata = self.metadata[self.metadata['sample'].isin(self.sample)]
        self.metadata['label'] = self.metadata['species'].astype('category').cat.codes
        self.labels = self.metadata.set_index('sample')['label'].to_dict()

    @property
    def n_classes(self):
        return len(np.unique(np.array(list(self.labels.values()))))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        sketch = self.sketches[idx]
        path = Path(sketch['file_str'])
        file_name = path.name
        sample = file_name[:-3]

        hv = decompress_hd_sketch(sketch)
        label = self.labels[sample]

        return torch.tensor(hv) / 255.0, torch.tensor(label, dtype=torch.long)