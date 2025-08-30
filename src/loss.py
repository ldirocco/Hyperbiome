import torch
import torch.nn as nn
import src.poincare_math as pmath

from modules import HypProjector

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
        P = self.projector(self.proxies_tan)  # mantiene la logica attuale

        # Calcolo della matrice di distanza iperbolica
        dist_mat = pmath.dist_matrix(X, P, self.c)  # distanza iperbolica

        # One-hot encoding dei target
        P_one_hot = torch.nn.functional.one_hot(T, num_classes=self.nb_classes).float()
        N_one_hot = 1 - P_one_hot

        # Selezione dei proxy positivi validi
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)

        pos_term = 0.0

        for p in with_pos_proxies:
            x_pos = P_one_hot[:, p].bool()
            pos_term += torch.log1p(torch.exp(dist_mat[x_pos, p])).sum()
        pos_term /= num_valid_proxies

        neg_term = 0.0
        for p in range(self.nb_classes):
            x_neg = N_one_hot[:, p].bool()
            neg_term += torch.log1p(torch.exp(-dist_mat[x_neg, p])).sum()
        neg_term /= self.nb_classes

        loss = pos_term + neg_term

        return loss

    def get_proxies(self):
        with torch.no_grad():
            return self.projector(self.proxies_tan)
