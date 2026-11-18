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


class MultiResSTFTLoss(nn.Module):
    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.resolutions = resolutions
        for n_fft, _ in resolutions:
            self.register_buffer(f"win_{n_fft}", torch.hann_window(n_fft), False)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # pred, tgt: [B, samples]. Spectral convergence + log-magnitude L1,
        # averaged over resolutions. Magnitudes only, so phase is free.
        loss = 0.0
        for n_fft, hop in self.resolutions:
            win = getattr(self, f"win_{n_fft}")
            p = torch.stft(pred, n_fft, hop, window=win, return_complex=True).abs()
            t = torch.stft(tgt, n_fft, hop, window=win, return_complex=True).abs()
            p, t = p.clamp(min=1e-5), t.clamp(min=1e-5)
            sc = torch.linalg.norm(t - p) / torch.linalg.norm(t)
            loss = loss + sc + F.l1_loss(p.log(), t.log())
        # pyrefly: ignore [bad-return]
        return loss / len(self.resolutions)


class WaveformLoss(nn.Module):
    # Renders a crop of the predicted mel with the frozen vocoder and compares
    # it with the clean audio, so the UNet is trained on what will be heard.
    CONTEXT = 8  # extra mel frames each side; their audio is discarded

    def __init__(self, generator: nn.Module, hop: int, frames: int, batch: int):
        super().__init__()
        self.generator = generator.requires_grad_(False)
        self.stft = MultiResSTFTLoss()
        self.hop = hop
        self.frames = frames
        self.batch = batch

    def forward(self, pred_mel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # pred_mel: [B, 1, n_mels, T], audio: [B, samples] aligned to the mel.
        ctx, hop = self.CONTEXT, self.hop
        T = pred_mel.shape[-1]
        frames = min(self.frames, T - 2 * ctx)
        a = int(torch.randint(ctx, T - frames - ctx + 1, (1,)))
        mel = pred_mel[: self.batch, 0, :, a - ctx : a + frames + ctx].float()
        with torch.autocast(device_type=mel.device.type, enabled=False):
            out = self.generator(mel)[:, 0, ctx * hop : (ctx + frames) * hop]
            # The mels are centred on i * hop while the vocoder expects frames
            # that start there, so its output lags the audio by half a hop.
            start = a * hop - hop // 2
            tgt = audio[: self.batch, start : start + frames * hop].float()
            return self.stft(out, tgt)
