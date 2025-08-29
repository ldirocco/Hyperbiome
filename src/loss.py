import torch
import torch.nn as nn
import poincare_math as pmath

from modules import HypProjector

# Appunti by Ciro:
#1. I proxy sono ottimizzati nello spazio tangente all’origine e poi mappati sulla palla con expmap0 durante il forward.
#   In questo modo 1) Nessun rischio che la proxy esca dalla palla 2) L’ottimizzazione rimane stabile su uno spazio Euclideo.
#2. La proxy anchor loss necessita si similarità: valori grandi per corrispondenze vicine e valori piccoli per corrispondenze lontane.
#   La formulazione stamdard usa cosine similarity. Possiamo fare lo stesso con distanza iperbolica

class HypProxyAnchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, c=1.0, mrg=0.1, alpha=32, clip_r=2.3, riemannian=True):
        super().__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.c = c
        self.mrg = mrg
        self.alpha = alpha

        init = torch.randn(nb_classes, sz_embed) * 0.01
        self.proxies_tan = nn.Parameter(init)
        self.projector = HypProjector(c=c, riemannian=riemannian, clip_r=clip_r)


    def forward(self, X, T):
        P = self.projector(self.proxies_tan)
        dist_mat = pmath.dist_matrix(X, P, c=self.c)

        sim = torch.exp(-self.alpha * (dist_mat - self.mrg))


        P_one_hot = torch.nn.functional.one_hot(T, num_classes=self.nb_classes).float()
        N_one_hot = 1 - P_one_hot

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)

        P_sim_sum = torch.where(P_one_hot == 1, sim, torch.zeros_like(sim)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, sim, torch.zeros_like(sim)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        loss = pos_term + neg_term
        return loss

    def get_proxies(self):
        with torch.no_grad():
            return self.projector(self.proxies_tan)
