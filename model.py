import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel

activation = get_kernel("kernels-community/activation")

class F0Encoder(nn.Module):
    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            activation.layers.Silu(),
            nn.Conv1d(32, 64, 5, padding=2),
            activation.layers.Silu(),
            nn.Conv1d(64, out_dim, 1)
        )

    def forward(self, f0: torch.Tensor, n_mels: int):
        # f0: [B, T]
        h = self.encoder(f0.unsqueeze(1)) # [B, C, T]
        h = h.unsqueeze(2).expand(-1, -1, n_mels, -1) # broadcast to mel bins
        return h # [B, C, n_mels, T]


class ShiftEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, out_dim),
            activation.layers.Silu(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, shift: torch.Tensor):
        return self.encoder(shift.unsqueeze(-1))
        

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], cond: [B, cond_dim]
        gb = self.proj(cond) # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta