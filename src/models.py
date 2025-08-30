import torch
import torch.nn as nn

from src.modules import HypProjector

class TransformerEmbedder(nn.Module):

    def __init__(self, input_dim=4096, patch_size=16, dim=128, depth=4, heads=4, mlp_dim=256):
        super().__init__()

        assert input_dim % patch_size == 0, "Input dimension must be divisible by patch size"

        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_cls_token = nn.Identity()


    def forward(self, x):
        # x: (batch_size, 4096)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_patches, -1)  # (B, num_patches, patch_size)
        x = self.patch_embed(x)  # (B, num_patches, dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, dim)
        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(x)  # (B, 1 + num_patches, dim)
        x = self.to_cls_token(x[:, 0])  # Use CLS token

        return x

class HypTransformerEmbedder(nn.Module):
    def __init__(self, input_dim=4096, patch_size=16, dim=128, depth=4, heads=4, mlp_dim=256, c=0.1):
        super().__init__()
        self.embedder = TransformerEmbedder(
            input_dim=input_dim,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )
        self.projector = HypProjector(c=c, riemannian=True, clip_r=2.3)

    def forward(self, x):
        x = self.embedder(x)
        x = self.projector(x)
        return x