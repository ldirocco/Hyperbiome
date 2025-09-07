import torch
import torch.nn as nn
import pandas as pd
import src.poincare_math as pmath

from src.modules import HypProjector




# Appunti by Ciro:
#1. I proxy sono ottimizzati nello spazio tangente all’origine e poi mappati sulla palla con expmap0 durante il forward.
#   In questo modo 1) Nessun rischio che la proxy esca dalla palla 2) L’ottimizzazione rimane stabile su uno spazio Euclideo.
#2. La proxy anchor loss necessita si similarità: valori grandi per corrispondenze vicine e valori piccoli per corrispondenze lontane.
#   La formulazione stamdard usa cosine similarity. Possiamo fare lo stesso con distanza iperbolica

class HypProxyAnchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, c=0.1, mrg=0.1, alpha=32, clip_r=2.3, riemannian=True):
        super().__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.c = c
        self.mrg = mrg
        self.alpha = alpha

        proxy_init = torch.randn(nb_classes, sz_embed) * 0.01
        self.proxies_tan = nn.Parameter(proxy_init)
        self.projector = HypProjector(c=c, riemannian=riemannian, clip_r=clip_r)


    def forward(self, X, T):
        # Proxies
        device=X.device
        P = self.projector(self.proxies_tan.to(device=device))  # mantiene la logica attuale

        # Calcolo della matrice di distanza iperbolica
        dist_mat = pmath.dist_matrix(X, P, self.c)  # distanza iperbolica

        # One-hot encoding dei target
        P_one_hot = torch.nn.functional.one_hot(T, num_classes=self.nb_classes).float()
        N_one_hot = 1 - P_one_hot

        valid_proxies_mask = (P_one_hot.sum(dim=0) != 0).float()  # [nb_classes]
        num_valid_proxies = valid_proxies_mask.sum().clamp(min=1.0)

        # Pos term: softplus(distanza) solo per coppie (x, proxy) positive
        pos_term = torch.sum(torch.nn.functional.softplus(dist_mat) * P_one_hot) / num_valid_proxies

        # --- NEGATIVE TERM ---
        # Neg term: softplus(-distanza) per le coppie negative
        neg_term = torch.sum(torch.nn.functional.softplus(-dist_mat) * N_one_hot) / self.nb_classes

        # Loss finale
        loss = pos_term + neg_term
        return loss


    def get_proxies(self):
        with torch.no_grad():
            return self.projector(self.proxies_tan)


class HypMultiProxyAnchor(nn.Module):
    def __init__(self, n_genus,n_species, sz_embed,metadata_path, c=0.1, mrg=0.1, alpha=32, clip_r=2.3, riemannian=True):
        super().__init__()
        self.n_genus = n_genus
        self.n_species = n_species

        self.sz_embed = sz_embed
        self.c = c
        self.mrg = mrg
        self.alpha = alpha

        self.projector = HypProjector(c=c, riemannian=riemannian, clip_r=clip_r)

        genus_proxy_init = torch.randn(n_genus, sz_embed) * 0.01
        self.genus_proxies_tan = nn.Parameter(genus_proxy_init)

        species_proxy_init = torch.randn(n_species, sz_embed) * 0.01
        self.species_proxies_tan = nn.Parameter(species_proxy_init)

        df = pd.read_csv(metadata_path, sep="\t")
        self.species_to_genus = dict(zip(df["Species_ID"], df["Genus_ID"]))


    def forward(self, X, T_species, T_genus):
        # Proxies
        device=X.device

        hyp_genus_proxies=self.projector(self.genus_proxies_tan.to(device=device))
        hyp_species_proxies=self.projector(self.species_proxies_tan.to(device=device))

        # ---- LOSS SPECIES ----
        # Calcolo della matrice di distanza iperbolica tra batteri e proxies delle specie
        dist_species = pmath.dist_matrix(X, hyp_species_proxies, self.c)  # distanza iperbolica

        P_one_hot = torch.nn.functional.one_hot(T_species, num_classes=self.n_species).float()
        N_one_hot = 1 - P_one_hot

        valid_species_proxies_mask = (P_one_hot.sum(dim=0) != 0).float()  # [nb_classes]
        num_valid_species_proxies = valid_species_proxies_mask.sum().clamp(min=1.0)

        pos_species_term = torch.sum(torch.nn.functional.softplus(dist_species) * P_one_hot) / num_valid_species_proxies
        neg_species_term = torch.sum(torch.nn.functional.softplus(-dist_species) * N_one_hot) / self.n_species

        loss_species = pos_species_term + neg_species_term


        # ---- LOSS GENUS (proxy specie ↔ proxy genere) ----

        # Prendo l'insieme di specie presenti nel batch
        unique_species, inv_idx = torch.unique(T_species, return_inverse=True)

        #Prendo in considerazione solamente le proxy corrispondenti alle specie in unique species
        species_proxies_unique = self.hyp_species_proxies[unique_species]

        #Calcolo la distanza tra il sottoinsieme di proxies delle species e le proxy dei genera
        dist_genus = pmath.dist_matrix(species_proxies_unique, hyp_genus_proxies, self.c)

        #Per ogni specie, calcoliamo il genus corrispondente
        genus_targets = torch.tensor([self.species_to_genus[s.item()] for s in unique_species])

        P_one_hot_genus = torch.nn.functional.one_hot(genus_targets, num_classes=60).float()
        N_one_hot_genus = 1 - P_one_hot_genus

        valid_genus_proxies_mask=(P_one_hot_genus.sum(dim=0) != 0).float()
        num_valid_genus_proxies = valid_genus_proxies_mask.sum().clamp(min=1.0)

        pos_genus_term=torch.sum(torch.nn.functional.softplus(dist_genus) * P_one_hot_genus) / num_valid_genus_proxies
        neg_genus_term=torch.sum(torch.nn.functional.softplus(-dist_genus) * N_one_hot_genus) / self.n_genus

        loss_genus = pos_genus_term + neg_genus_term

        # ---- LOSS TOTALE ----
        loss = loss_species + loss_genus
        return loss


    def get_species_proxies(self):
        with torch.no_grad():
            return self.projector(self.species_proxies_tan)

    def get_genus_proxies(self):
        with torch.no_grad():
            return self.projector(self.genus_proxies_tan)
