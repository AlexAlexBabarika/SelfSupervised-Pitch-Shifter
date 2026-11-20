import pyloudnorm
import pyworld
import librosa
import torchaudio
import torch
import torchcrepe
import numpy as np
import math
from typing import Generator
from config import AudioConfig
from pathlib import Path

cfg = AudioConfig()

SUPPORTED_SRS = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
CREPE_MODEL_CAPACITIES = ["tiny", "full"]
CLIP_LEN = int(cfg.clip_seconds * cfg.target_sr)
# Stored segments are longer than a training clip so the loader can crop at
# random offsets.
SEG_LEN = int(cfg.segment_seconds * cfg.target_sr)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

meter = pyloudnorm.Meter(rate=cfg.target_sr)
_MEL_TRANSFORM = None


def _mel_transform() -> torchaudio.transforms.MelSpectrogram:
    # Built on first use: data-loader workers import this module, and a
    # module-level .to("cuda") would open a CUDA context in every one of them.
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
        _MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.target_sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=1.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        ).to(DEVICE)
    return _MEL_TRANSFORM


def load_audio_file(file_path: str) -> tuple[np.ndarray, int | float]:
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def resample_audio_to_target_sr(
    audio: np.ndarray, original_sr: float | int
) -> np.ndarray:
    if cfg.target_sr not in SUPPORTED_SRS:
        raise ValueError(
            f"The target rate of {cfg.target_sr} isn't supported. Supported rates:\n{SUPPORTED_SRS}"
        )
    return librosa.resample(audio, orig_sr=original_sr, target_sr=cfg.target_sr)


def normalization_gain(audio: np.ndarray) -> float:
    gain = 1.0
    try:
        loud = meter.integrated_loudness(audio)
        if math.isfinite(loud):
            gain = 10.0 ** ((cfg.target_lufs - loud) / 20.0)
    except Exception:
        pass
    peak = float(np.max(np.abs(audio))) * gain + 1e-9
    if peak > 0.99:
        gain *= 0.99 / peak
    return gain


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    return (audio * normalization_gain(audio)).astype(np.float32)


def segment_audio(audio: np.ndarray) -> Generator[tuple[np.ndarray, int]]:
    n = len(audio)
    for start in range(0, n, SEG_LEN):
        clip = audio[start : start + SEG_LEN]
        real_len = len(clip)
        if real_len < CLIP_LEN // 2:
            continue
        if real_len < SEG_LEN:
            clip = np.pad(clip, (0, SEG_LEN - real_len))
        yield clip, real_len


def compute_mel_spectrogram(clips: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(clips).float().to(DEVICE)
    m = _mel_transform()(t).cpu().numpy()
    return np.log(np.maximum(m, 1e-5)).astype(np.float32)


def f0_to_feature(
    f0_hz: torch.Tensor,
    conf: torch.Tensor,
    fmax: float = cfg.fmax,
) -> torch.Tensor:
    # f0_hz, conf: [..., T]. Returns [..., 2, T] = (log2(f0+1), voiced_mask)
    # with voicing dropped when the fundamental leaves the analysable band.
    voiced = (conf > cfg.voiced_threshold).to(f0_hz.dtype)
    in_range = (f0_hz < (fmax / 2.0)).to(f0_hz.dtype)
    voiced = voiced * in_range
    f0_norm = torch.log2(f0_hz.clamp(min=0.0) + 1.0)
    return torch.stack([f0_norm, voiced], dim=-2)


def world_pitch_shift(
    audio: np.ndarray,
    f0_hz: np.ndarray,
    conf: np.ndarray,
    semis: float,
    sr: int = cfg.target_sr,
    formant_semis: float = 0.0,
) -> np.ndarray:
    # audio: [samples] @ sr, f0_hz/conf: [T] on the mel frame grid (hop_length
    # samples at target_sr, whatever sr is). WORLD resynthesis with the F0
    # track scaled and the spectral envelope kept, so formants stay put, or
    # moved by formant_semis.
    x = np.ascontiguousarray(audio, dtype=np.float64)
    f0 = np.where(conf > cfg.voiced_threshold, f0_hz, 0.0).astype(np.float64)
    t = np.arange(len(f0), dtype=np.float64) * cfg.hop_length / cfg.target_sr
    sp = pyworld.cheaptrick(x, f0, t, sr)
    ap = pyworld.d4c(x, f0, t, sr)
    if abs(formant_semis) > 1e-6:
        sp = _warp_frequency(sp, sr, formant_semis)
        ap = _warp_frequency(ap, sr, formant_semis)
    y = pyworld.synthesize(
        f0 * 2.0 ** (semis / 12.0),
        sp,
        ap,
        sr,
        1000.0 * cfg.hop_length / cfg.target_sr,
    )
    n = len(x)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def _warp_frequency(frames: np.ndarray, sr: int, semis: float) -> np.ndarray:
    # frames: [T, bins] over 0..sr/2. Moves every feature up by `semis`: the
    # value at f lands at f * 2**(semis / 12).
    r = 2.0 ** (semis / 12.0)
    freqs = np.linspace(0.0, sr / 2.0, frames.shape[1])
    return np.ascontiguousarray(
        np.stack([np.interp(freqs / r, freqs, row) for row in frames])
    )


def _decode_f0(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # probs: [B, 360, T]. Argmax bin refined by a probability-weighted average
    # over its neighbours. torchcrepe's own argmax decoder snaps to 20-cent
    # bins and then adds random dither, which makes the track jitter.
    bins = probs.argmax(dim=1)
    offsets = torch.arange(-4, 5, device=probs.device).view(1, -1, 1)
    idx = bins.unsqueeze(1) + offsets
    inside = (idx >= 0) & (idx < probs.shape[1])
    idx = idx.clamp(0, probs.shape[1] - 1)
    w = probs.gather(1, idx).clamp(min=0.0) * inside
    cents = torchcrepe.CENTS_PER_BIN * idx + 1997.3794084376191
    cents = (w * cents).sum(1) / w.sum(1).clamp(min=1e-8)
    return bins, torchcrepe.convert.cents_to_frequency(cents)


def extract_f0(clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cfg.crepe_model_capacity not in CREPE_MODEL_CAPACITIES:
        raise ValueError(
            f"Model capacity {cfg.crepe_model_capacity} isn't supported by torchcrepe. Supported capacities:\n{CREPE_MODEL_CAPACITIES}"
        )

    audio_t = torch.from_numpy(clips).float().to(DEVICE)
    audio_16k = torchaudio.functional.resample(audio_t, cfg.target_sr, 16000)
    # CREPE frame interval is 186/16000 ≈ 11.625 ms; mel frame interval is 256/22050 ≈ 11.610 ms. The np.interp later masks it
    # Across 220 frames it's gonna be around 3ms. Has to be examined
    hop = round(16000 * cfg.hop_length / cfg.target_sr)

    pitch, periodicity = torchcrepe.predict(
        audio_16k,
        sample_rate=16000,
        hop_length=hop,
        model=cfg.crepe_model_capacity,
        batch_size=512,
        device=DEVICE,
        decoder=_decode_f0,
        return_periodicity=True,
    )
    pitch = torchcrepe.filter.median(pitch, 3).cpu().numpy()
    periodicity = torchcrepe.filter.median(periodicity, 3).cpu().numpy()
    if DEVICE == "mps":
        # MPS keeps several GB cached after CREPE; hand it back.
        torch.mps.empty_cache()

    return pitch.astype(np.float32), periodicity.astype(np.float32)


def process_and_save(
    audio: np.ndarray, src_sr: int, out_dir: Path, base_id: str, budget: dict
) -> float:
    if budget["remaining_s"] <= 0:
        return 0.0
    if src_sr != cfg.target_sr:
        audio = resample_audio_to_target_sr(audio, src_sr)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) < cfg.target_sr * 0.5:
        return 0.0
    audio = normalize_audio(audio)

    # Collect clips up to the budget, then run mel + CREPE on the whole batch
    clips: list[np.ndarray] = []
    real_lens: list[int] = []
    remaining = budget["remaining_s"]
    for clip, real_len in segment_audio(audio):
        if remaining <= 0:
            break
        clips.append(clip)
        real_lens.append(real_len)
        remaining -= real_len / cfg.target_sr
    if not clips:
        return 0.0

    batch = np.stack(clips, axis=0)
    f0s, confs = extract_f0(batch)  # (B, T_f0)
    n_frames = 1 + SEG_LEN // cfg.hop_length

    written = 0.0
    for i, (clip, real_len, f0, conf) in enumerate(zip(clips, real_lens, f0s, confs)):
        if f0.shape[0] != n_frames:
            # Nearest-neighbor: linear interp would smear CREPE's 0-Hz unvoiced
            # frames into bogus low pitches across voiced -> unvoiced transitions.
            idx = np.round(np.linspace(0, f0.shape[0] - 1, n_frames)).astype(np.int64)
            f0 = f0[idx]
            conf = conf[idx]
        out_path = out_dir / f"{base_id}_{i:04d}.npz"
        np.savez(
            out_path,
            audio=clip[:real_len].astype(np.float32),
            f0=f0,
            conf=conf,
            real_len=real_len,
        )
        seconds = real_len / cfg.target_sr
        written += seconds
        budget["remaining_s"] -= seconds
    return written
