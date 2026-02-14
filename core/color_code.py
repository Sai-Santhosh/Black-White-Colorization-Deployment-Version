"""Color space normalization for Lab-based colorization."""
import torch
from torch import nn


class ColorCode(nn.Module):
    """Base class for L/ab channel normalization in Lab color space."""

    def __init__(self) -> None:
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab * self.ab_norm
