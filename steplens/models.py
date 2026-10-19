"""Error-recognition heads, ported from CaptainCook4D/error_recognition.

* MLP        1024 -> 512 -> 1 on each 1 s feature row  (paper variant V1)
* ErFormer   one TransformerEncoderLayer over the step's rows, then the MLP
             (paper variant V2; V3 is the same head on concatenated ImageBind
             modalities, which it averages with fixed weights before the MLP)

Both take an unbatched (T, D) tensor for one step and return (T, 1) logits.
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 512, output_size: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


class ErFormer(nn.Module):
    def __init__(self, input_dim: int, n_modalities: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.n_modalities = n_modalities
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.step_encoder = nn.TransformerEncoder(layer, num_layers=1)
        per_mod = input_dim // n_modalities
        self.decoder = MLP(per_mod, 512, 1)
        # fixed modality weights from the official code (video, audio, text, depth)
        self.weights = {1: [1.0], 2: [0.65, 0.35], 3: [0.4, 0.3, 0.3], 4: [0.25, 0.25, 0.25, 0.25]}[n_modalities]

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        enc = self.step_encoder(x)  # unbatched (T, D)
        if self.n_modalities == 1:
            fused = enc
        else:
            d = self.input_dim // self.n_modalities
            parts = [enc[:, i * d:(i + 1) * d] for i in range(self.n_modalities)]
            fused = sum(w * p for w, p in zip(self.weights, parts))
        return self.decoder(fused)


def build_model(variant: str, input_dim: int, n_modalities: int = 1) -> nn.Module:
    variant = variant.lower()
    if variant == "mlp":
        return MLP(input_dim, 512, 1)
    if variant == "transformer":
        return ErFormer(input_dim, n_modalities)
    raise ValueError(f"unknown variant {variant}")
