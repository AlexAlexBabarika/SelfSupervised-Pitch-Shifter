import json
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

from inference.hifigan.config import V1_CONFIG
from inference.hifigan.models import Generator


class Vocoder:
    # Shared interface: log-mel in, audio out. Subclasses set `generator`,
    # `device`, `sample_rate` and `hop`.
    generator: torch.nn.Module
    device: torch.device
    sample_rate: int
    hop: int

    @torch.inference_mode()
    def mel_to_audio(self, log_mel: torch.Tensor) -> torch.Tensor:
        # log_mel: [B, n_mels, T] in natural-log magnitude space.
        # Returns: [B, samples] float32 audio, peak-clamped to [-1, 1].
        x = log_mel.to(self.device)
        audio = self.generator(x).squeeze(1).clamp(-1.0, 1.0).float().cpu()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return audio

    @torch.inference_mode()
    def mel_to_audio_long(
        self, log_mel: torch.Tensor, chunk: int = 1024, context: int = 64
    ) -> torch.Tensor:
        # log_mel: [n_mels, T] of any length. Vocodes in chunks that see
        # `context` extra frames on each side and keeps only the centre, so
        # the result matches one pass over the whole mel without its memory.
        T = log_mel.shape[-1]
        pieces = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            a, b = max(0, start - context), min(T, end + context)
            audio = self.mel_to_audio(log_mel[None, :, a:b])[0]
            lo = (start - a) * self.hop
            pieces.append(audio[lo : lo + (end - start) * self.hop].cpu())
        # Our mels are centred on i * hop, the vocoder expects frames that
        # start there, so its output lags by half a hop.
        return torch.cat(pieces)[self.hop // 2 :]


class HiFiGANVocoder(Vocoder):
    def __init__(
        self,
        repo_id: str,
        device: torch.device,
        ckpt_filename: str = "g_02500000",
        config_filename: str | None = "config.json",
    ):
        if config_filename is not None:
            cfg_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = dict(V1_CONFIG)
        h = SimpleNamespace(**cfg)

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.generator = Generator(h).to(device)
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.device = device
        self.sample_rate = int(getattr(h, "sampling_rate", V1_CONFIG["sampling_rate"]))
        self.hop = int(getattr(h, "hop_size", V1_CONFIG["hop_size"]))


class BigVGANVocoder(Vocoder):
    def __init__(self, repo_id: str, device: torch.device):
        from inference.bigvgan.bigvgan import BigVGAN
        from inference.bigvgan.env import AttrDict

        # Loaded by hand: BigVGAN.from_pretrained breaks on current
        # huggingface_hub versions.
        with open(hf_hub_download(repo_id=repo_id, filename="config.json")) as f:
            generator = BigVGAN(AttrDict(json.load(f)))
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="bigvgan_generator.pt")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        generator.load_state_dict(state["generator"])
        generator.remove_weight_norm()
        self.generator = generator.eval().to(device)
        self.device = device
        self.sample_rate = int(generator.h.sampling_rate)
        self.hop = int(generator.h.hop_size)


def load_vocoder(repo_id: str, device: torch.device) -> Vocoder:
    if "bigvgan" in repo_id.lower():
        return BigVGANVocoder(repo_id, device)
    return HiFiGANVocoder(repo_id, device)
