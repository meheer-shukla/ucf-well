"""Stress classification head with Cross-Modal HRV Transfer (CMHT).

Used by scripts/step5_train_cmht.py.

StressNet : small MLP encoder + linear classifier
coral_loss: Deep CORAL (Sun et al. 2016) — squared Frobenius distance
            between source and target feature covariances
jitter_batch: rPPG-realistic IBI jitter augmentation on source features
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class StressNet(nn.Module):
    def __init__(self, in_dim: int = 14, hidden: int = 64,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


# ----------------------------------------------------------------------
# CORAL loss
# ----------------------------------------------------------------------
def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared Frobenius distance between source and target
    covariance matrices.

    Sun, Feng, Saenko, "Return of Frustratingly Easy Domain Adaptation",
    AAAI 2016.
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    if ns < 2 or nt < 2:
        return source.new_zeros(1).squeeze()

    xm = source - source.mean(0, keepdim=True)
    ym = target - target.mean(0, keepdim=True)
    cs = (xm.t() @ xm) / (ns - 1)
    ct = (ym.t() @ ym) / (nt - 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


# ----------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------
def rppg_ibi_jitter(ibi: torch.Tensor, jitter_std: float = 0.025,
                    dropout_p: float = 0.05) -> torch.Tensor:
    """Simulate rPPG peak-detection error on contact-derived IBI features.

    Two effects:
      (i)  additive timing jitter  ~ N(0, jitter_std^2)
      (ii) random beat dropout with probability dropout_p
    """
    out = ibi.clone()
    out = out + torch.randn_like(out) * jitter_std
    if dropout_p > 0:
        keep = (torch.rand_like(out) > dropout_p).float()
        out = out * keep + out.mean() * (1 - keep)
    return out.clamp_min(0.25)