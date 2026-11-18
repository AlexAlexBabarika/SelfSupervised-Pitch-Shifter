import math

import librosa
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from audio_preprocessing import (
    compute_mel_spectrogram,
    extract_f0,
    f0_to_feature,
    normalization_gain,
    world_pitch_shift,
)
from config import AudioConfig
from inference.stitch import overlap_add
from inference.vocoder import HiFiGANVocoder
from model import PitchUNet


def shift_clip_batch(
    batch_np: np.ndarray,
    semis: float,
    unet: PitchUNet | None,
    vocoder: HiFiGANVocoder,
    device: torch.device,
) -> np.ndarray:
    # batch_np: [B, clip_len] @ AudioConfig.target_sr, loudness-normalised.
    # Returns [B, clip_len]. unet=None vocodes the WORLD output directly.
    cfg = AudioConfig()
    B, clip_len = batch_np.shape

    f0_np, conf_np = extract_f0(batch_np)  # [B, T_f0]

    T_mel = 1 + clip_len // cfg.hop_length
    T_f0 = f0_np.shape[-1]
    if T_f0 != T_mel:
        idx = np.round(np.linspace(0, T_f0 - 1, T_mel)).astype(np.int64)
        f0_np = f0_np[:, idx]
        conf_np = conf_np[:, idx]

    # WORLD does the shift; the UNet only cleans up its artifacts.
    if abs(semis) > 1e-4:
        batch_np = np.stack(
            [
                world_pitch_shift(c, f, p, semis)
                for c, f, p in zip(batch_np, f0_np, conf_np)
            ]
        )
        f0_np = f0_np * 2.0 ** (semis / 12.0)
    mel_np = compute_mel_spectrogram(batch_np)  # [B, n_mels, T_mel]

    mel = torch.from_numpy(mel_np).to(device)  # [B, n_mels, T]
    with torch.inference_mode():
        if unet is not None:
            f0_hz = torch.from_numpy(f0_np).to(device)
            conf = torch.from_numpy(conf_np).to(device)
            f0_feat = f0_to_feature(f0_hz, conf, fmax=cfg.fmax)  # [B, 2, T]
            shift = torch.full((B,), float(semis), device=device)
            cond_mask = torch.ones((B,), device=device)
            mel = unet(mel.unsqueeze(1), f0_feat, shift, cond_mask).squeeze(1)
        audio_out = vocoder.mel_to_audio(mel)  # [B, S]

    a = audio_out.cpu().numpy()[:, vocoder.hop // 2 :]  # undo the half-hop lag
    out = np.zeros((B, clip_len), dtype=np.float32)
    L = min(a.shape[-1], clip_len)
    out[:, :L] = a[:, :L]
    return out


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if x.shape[0] >= n else np.pad(x, (0, n - x.shape[0]))


def _refine_mel(
    mel: np.ndarray,
    f0_hz: np.ndarray,
    conf: np.ndarray,
    semis: float,
    unet: PitchUNet,
    device: torch.device,
    batch_size: int,
    overlap: float,
    verbose: bool,
) -> np.ndarray:
    # mel: [n_mels, T], f0_hz/conf: [T]. Runs the UNet over training-sized
    # windows and cross-fades the predicted mels.
    cfg = AudioConfig()
    T = mel.shape[-1]
    win = 1 + int(cfg.clip_seconds * cfg.target_sr) // cfg.hop_length
    hop = max(1, int(win * (1.0 - overlap)))
    if T <= win:
        win, n_win = T, 1
    else:
        n_win = math.ceil((T - win) / hop) + 1
    pad = hop * (n_win - 1) + win - T
    mel = np.pad(mel, ((0, 0), (0, pad)), constant_values=math.log(1e-5))
    f0_hz = np.pad(f0_hz, (0, pad))
    conf = np.pad(conf, (0, pad))

    starts = [i * hop for i in range(n_win)]
    out = np.zeros((n_win, mel.shape[0], win), dtype=np.float32)
    iter_range = range(0, n_win, batch_size)
    if verbose:
        iter_range = tqdm(iter_range, desc="UNet")
    for i in iter_range:
        idx = starts[i : i + batch_size]
        m = torch.from_numpy(np.stack([mel[:, s : s + win] for s in idx])).to(device)
        f = torch.from_numpy(np.stack([f0_hz[s : s + win] for s in idx])).to(device)
        c = torch.from_numpy(np.stack([conf[s : s + win] for s in idx])).to(device)
        f0_feat = f0_to_feature(f, c, fmax=cfg.fmax)  # [B, 2, win]
        shift = torch.full((len(idx),), float(semis), device=device)
        with torch.inference_mode():
            pred = unet(m.unsqueeze(1), f0_feat, shift, torch.ones_like(shift))
        out[i : i + len(idx)] = pred.squeeze(1).float().cpu().numpy()

    return overlap_add(out, hop=hop, clip_len=win)[:, :T]


def _lowpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    sos = butter(8, cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def shift_audio_through_model(
    mono_audio: np.ndarray,
    sr: int,
    semis: float,
    unet: PitchUNet,
    vocoder: HiFiGANVocoder,
    device: torch.device,
    batch_size: int = 8,
    overlap: float = 0.25,
    verbose: bool = False,
) -> np.ndarray:
    # mono_audio: [samples] @ sr. Returns the shifted waveform at the same
    # rate and length. The model works at AudioConfig.target_sr and up to
    # fmax; the band above fmax is taken from the WORLD shift at full rate.
    cfg = AudioConfig()
    n = int(mono_audio.shape[-1])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    mono_audio = mono_audio.astype(np.float32, copy=False)

    x = mono_audio
    if sr != cfg.target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=cfg.target_sr)

    # Training clips are loudness-normalised; match that here and undo it on
    # the way out.
    gain = normalization_gain(x)
    x = x * gain

    # F0 and WORLD run once over the whole signal, so there are no clip edges.
    n_frames = 1 + x.shape[0] // cfg.hop_length
    f0, conf = (a[0] for a in extract_f0(x[None]))
    if f0.shape[0] != n_frames:
        idx = np.round(np.linspace(0, f0.shape[0] - 1, n_frames)).astype(np.int64)
        f0, conf = f0[idx], conf[idx]

    shifted = world_pitch_shift(x, f0, conf, semis) if abs(semis) > 1e-4 else x
    mel = compute_mel_spectrogram(shifted[None])[0]  # [n_mels, T]
    mel = _refine_mel(
        mel,
        f0 * 2.0 ** (semis / 12.0),
        conf,
        semis,
        unet,
        device,
        batch_size,
        overlap,
        verbose,
    )

    # One vocoder pass over the stitched mel: no audio cross-fades.
    out = vocoder.mel_to_audio_long(torch.from_numpy(mel)).numpy()
    out = _fit(out, x.shape[0])
    if sr != cfg.target_sr:
        out = _fit(librosa.resample(out, orig_sr=cfg.target_sr, target_sr=sr), n)

    if sr / 2 > cfg.fmax:
        full = mono_audio * gain
        if abs(semis) > 1e-4:
            full = world_pitch_shift(full, f0, conf, semis, sr=sr)
        out = _lowpass(out, cfg.fmax, sr) + (full - _lowpass(full, cfg.fmax, sr))

    return (out / gain).astype(np.float32)
