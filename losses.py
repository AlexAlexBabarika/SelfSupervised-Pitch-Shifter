import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResMelLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for s in self.scales:
            d = diff if s == 1 else F.avg_pool2d(diff, kernel_size=s)
            loss = loss + d.abs().mean()
        # pyrefly: ignore [bad-return]
        return loss / len(self.scales)


class DeltaLoss(nn.Module):
    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        # L1 on the frequency and time differences: penalises blurred
        # harmonics and onsets that plain L1 lets through.
        d_freq = diff[..., 1:, :] - diff[..., :-1, :]
        d_time = diff[..., :, 1:] - diff[..., :, :-1]
        return 0.5 * (d_freq.abs().mean() + d_time.abs().mean())


class TotalLoss(nn.Module):
    def __init__(self, w_l1: float = 1.0, w_mr: float = 0.5, w_delta: float = 0.5):
        super().__init__()
        self.mr = MultiResMelLoss()
        self.delta = DeltaLoss()
        self.w_l1 = w_l1
        self.w_mr = w_mr
        self.w_delta = w_delta

    def forward(self, pred, tgt, mask):
        # mask: [B, 1, 1, T], 1 on real frames. Every term is linear in the
        # masked difference, so dividing by the kept fraction averages over
        # real frames only.
        diff = (pred - tgt) * mask
        kept = mask.mean().clamp(min=1e-8)
        l1 = diff.abs().mean() / kept
        mr = self.mr(diff) / kept
        delta = self.delta(diff) / kept
        total = self.w_l1 * l1 + self.w_mr * mr + self.w_delta * delta
        return total, {"l1": l1.item(), "mr": mr.item(), "delta": delta.item()}
