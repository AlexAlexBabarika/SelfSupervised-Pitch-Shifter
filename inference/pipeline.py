import numpy as np
import torch
from tqdm import tqdm

from audio_preprocessing import compute_mel_spectrogram, extract_f0, f0_to_feature
from config import AudioConfig
from inference.stitch import overlap_add
from inference.vocoder import HiFiGANVocoder
from model import PitchUNet


def shift_audio_through_model(
    mono_audio: np.ndarray,
    semis: float,
    unet: PitchUNet,
    vocoder: HiFiGANVocoder,
    device: torch.device,
    batch_size: int = 8,
    overlap: float = 0.25,
    verbose: bool = False,
) -> np.ndarray:
    # mono_audio: [samples] @ AudioConfig.target_sr. Returns shifted waveform
    # at the same rate, trimmed to the original length.
    cfg = AudioConfig()
    sr = cfg.target_sr
    clip_len = int(cfg.clip_seconds * sr)
    hop = max(1, int(clip_len * (1.0 - overlap)))

    n = int(mono_audio.shape[-1])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    starts = list(range(0, n, hop)) or [0]
    clips = np.zeros((len(starts), clip_len), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = mono_audio[s : s + clip_len].astype(np.float32, copy=False)
        clips[i, : seg.shape[0]] = seg

    out_clips = np.zeros_like(clips)
    N = clips.shape[0]
    iter_range = range(0, N, batch_size)
    if verbose:
        iter_range = tqdm(
            iter_range, desc="UNet+vocoder", total=(N + batch_size - 1) // batch_size
        )

    for i in iter_range:
        batch_np = clips[i : i + batch_size]
        B = batch_np.shape[0]

        mel_np = compute_mel_spectrogram(batch_np)  # [B, n_mels, T_mel]
        f0_np, conf_np = extract_f0(batch_np)  # [B, T_f0]

        T_mel = mel_np.shape[-1]
        T_f0 = f0_np.shape[-1]
        if T_f0 != T_mel:
            idx = np.round(np.linspace(0, T_f0 - 1, T_mel)).astype(np.int64)
            f0_np = f0_np[:, idx]
            conf_np = conf_np[:, idx]

        mel = torch.from_numpy(mel_np).to(device)  # [B, n_mels, T]
        f0_hz = torch.from_numpy(f0_np).to(device)
        conf = torch.from_numpy(conf_np).to(device)
        f0_feat = f0_to_feature(f0_hz, conf, fmax=cfg.fmax)  # [B, 2, T]

        shift = torch.full((B,), float(semis), device=device)
        cond_mask = torch.ones((B,), device=device)

        with torch.inference_mode():
            mel_out = unet(
                mel.unsqueeze(1), f0_feat, shift, cond_mask
            )  # [B, 1, n_mels, T]
            audio_out = vocoder.mel_to_audio(mel_out.squeeze(1))  # [B, S]

        a = audio_out.cpu().numpy()
        L = a.shape[-1]
        if L >= clip_len:
            out_clips[i : i + B] = a[:, :clip_len]
        else:
            out_clips[i : i + B, :L] = a

    if N == 1:
        stitched = out_clips[0]
    else:
        stitched = overlap_add(out_clips, hop=hop, clip_len=clip_len)

    if stitched.shape[0] >= n:
        stitched = stitched[:n]
    else:
        stitched = np.pad(stitched, (0, n - stitched.shape[0]))
    return stitched.astype(np.float32)
