import back.src.poincare_math as pmath
import torch

from torch import nn


class HypProjector(nn.Module):

    def __init__(self, c,riemannian=True,clip_r=2.3):
        super().__init__()
        self.c = c
        self.riemannian=pmath.RiemannianGradient
        self.riemannian.c=c
        self.clip_r=clip_r

        if riemannian:
            self.grad_fix= lambda x: self.riemannian(x).apply(x)
        else:
            self.grad_fix= lambda x: x


    def forward(self, x):

        # expmap0 è la mappa geometrica.
        # project è un “safety check” perch+ corregge eventuali sforamenti numerici.
        # Infatti, a causa di errori floating-point (es. con float32), può succedere che il risultato dell’expmap0
        # cada leggermente fuori dalla palla

        x_norm=torch.norm(x, dim=1, keepdim=True)+1e-5
        fac=torch.minimum(
            torch.ones_like(x_norm),
            self.clip_r/x_norm
        )
        x=x*fac

        return self.grad_fix(pmath.project(pmath.expmap0(x,c=self.c), c=self.c))

