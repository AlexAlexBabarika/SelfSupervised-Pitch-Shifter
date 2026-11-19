import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import torch
from parselmouth.praat import call
from scipy.fft import dct
from tqdm import tqdm

from audio_preprocessing import (
    CLIP_LEN,
    compute_mel_spectrogram,
    extract_f0,
    world_pitch_shift,
)
from config import AudioConfig, TrainConfig
from data import PitchDataset
from inference.checkpoint import find_ckpt, load_unet
from inference.pipeline import shift_clip_batch
from inference.rubberband import rubberband_shift
from inference.vocoder import load_vocoder

cfg = AudioConfig()
SYSTEMS = [
    "librosa",
    "rubberband",
    "psola",
    "world",
    "world+vocoder",
    "world+unet+vocoder",
    "vocoder only",
]
# Stricter than the training threshold: only score frames CREPE is sure about.
EVAL_VOICED = 0.5
GROSS_CENTS = 50.0


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _f0(clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, conf = extract_f0(clips)
    T = 1 + clips.shape[-1] // cfg.hop_length
    if f0.shape[-1] != T:
        idx = np.round(np.linspace(0, f0.shape[-1] - 1, T)).astype(np.int64)
        f0, conf = f0[:, idx], conf[:, idx]
    return f0, conf


def _envelope(log_mel: np.ndarray, width: int = 9) -> np.ndarray:
    # Moving average over mel bins: keeps formants, smooths out harmonics.
    kernel = np.ones(width) / width
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"), -2, log_mel)


def _mcd(mel_a: np.ndarray, mel_b: np.ndarray, n_coef: int = 13) -> np.ndarray:
    # Mel-cepstral distortion in dB per frame, from 80-bin log-mels, c0 excluded.
    ca = dct(mel_a, type=2, axis=-2, norm="ortho")[..., 1 : n_coef + 1, :]
    cb = dct(mel_b, type=2, axis=-2, norm="ortho")[..., 1 : n_coef + 1, :]
    return (10.0 / np.log(10.0)) * np.sqrt(2.0 * ((ca - cb) ** 2).sum(-2))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = x[:n] if x.shape[0] >= n else np.pad(x, (0, n - x.shape[0]))
    return x.astype(np.float32)


def _rubberband(clip: np.ndarray, s: float) -> np.ndarray:
    # Rubber Band's R3 engine with full formant preservation.
    return rubberband_shift(clip, cfg.target_sr, s)


def _psola(clip: np.ndarray, s: float) -> np.ndarray:
    # Praat's overlap-add (PSOLA) resynthesis with the pitch tier scaled.
    snd = parselmouth.Sound(clip.astype(np.float64), sampling_frequency=cfg.target_sr)
    manip = call(snd, "To Manipulation", 0.01, 75, 800)
    tier = call(manip, "Extract pitch tier")
    call(tier, "Multiply frequencies", snd.xmin, snd.xmax, 2.0 ** (s / 12.0))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    return _fit(out.values[0], clip.shape[0])


def make_systems(unet, vocoder, device):
    def librosa_shift(clips, s):
        return np.stack(
            [librosa.effects.pitch_shift(c, sr=cfg.target_sr, n_steps=s) for c in clips]
        )

    def world(clips, s):
        f0, conf = _f0(clips)
        return np.stack(
            [world_pitch_shift(c, f, p, s) for c, f, p in zip(clips, f0, conf)]
        )

    return {
        "librosa": librosa_shift,
        "rubberband": lambda c, s: np.stack([_rubberband(x, s) for x in c]),
        "psola": lambda c, s: np.stack([_psola(x, s) for x in c]),
        "world": world,
        "world+vocoder": lambda c, s: shift_clip_batch(c, s, None, vocoder, device),
        "world+unet+vocoder": lambda c, s: shift_clip_batch(
            c, s, unet, vocoder, device
        ),
        # No shift at all: what the vocoder alone costs. Only meaningful in
        # the round-trip columns.
        "vocoder only": lambda c, s: shift_clip_batch(c, 0.0, None, vocoder, device),
    }


def score_batch(clips, f0_src, conf_src, mel_src, shift, system) -> dict[str, list]:
    out: dict[str, list] = {
        k: [] for k in ("cents", "gross", "kept", "env", "rt_l1", "rt_mcd")
    }

    # Single shift: no ground truth, so score pitch accuracy and how far the
    # spectral envelope moved from the source.
    y = system(clips, shift)
    f0_y, conf_y = _f0(y)
    mel_y = compute_mel_spectrogram(y)
    src_voiced = conf_src > EVAL_VOICED
    both = src_voiced & (conf_y > EVAL_VOICED)
    target = f0_src * 2.0 ** (shift / 12.0)
    cents = 1200.0 * np.abs(np.log2(f0_y[both] / target[both]))
    out["cents"].extend(cents.tolist())
    out["gross"].extend((cents > GROSS_CENTS).tolist())
    out["kept"].extend(both[src_voiced].tolist())
    env = np.abs(_envelope(mel_y) - _envelope(mel_src)).mean(-2)
    out["env"].extend(env[src_voiced].tolist())

    # Round trip: shift there and back, compare with the original clip on
    # its voiced frames.
    back = system(y, -shift)
    mel_back = compute_mel_spectrogram(back)
    out["rt_l1"].extend(np.abs(mel_back - mel_src).mean(-2)[src_voiced].tolist())
    out["rt_mcd"].extend(_mcd(mel_back, mel_src)[src_voiced].tolist())
    return out


def summarise(acc: dict[str, list]) -> dict[str, float]:
    def mean(k):
        return float(np.mean(acc[k])) if acc[k] else float("nan")

    return {
        "f0_median_cents": float(np.median(acc["cents"]))
        if acc["cents"]
        else float("nan"),
        "f0_gross_pct": 100.0 * mean("gross"),
        "voiced_kept_pct": 100.0 * mean("kept"),
        "envelope_dist": mean("env"),
        "roundtrip_mel_l1": mean("rt_l1"),
        "roundtrip_mcd_db": mean("rt_mcd"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Score the shifter against DSP baselines on held-out clips."
    )
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--shifts", default="-6,-3,-1,1,3,6")
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--vocoder-repo", default=TrainConfig().vocoder_repo)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out", type=Path, default=Path("eval_results.json"))
    args = p.parse_args(argv)

    device = _resolve_device(args.device)
    ckpt = args.ckpt or find_ckpt(Path("./checkpoints"))
    if ckpt is None:
        print("No checkpoint found and --ckpt not given.", file=sys.stderr)
        return 2
    unet = load_unet(ckpt, device, use_ema=not args.no_ema)
    vocoder = load_vocoder(args.vocoder_repo, device)
    systems = make_systems(unet, vocoder, device)
    names = list(SYSTEMS)
    if shutil.which("rubberband") is None:
        print("rubberband CLI not found; skipping that baseline.", file=sys.stderr)
        names.remove("rubberband")

    files = []
    for f in PitchDataset("val").files:
        with np.load(f) as d:
            if int(d["real_len"]) >= CLIP_LEN:
                files.append(f)
    random.Random(0).shuffle(files)
    files = files[: args.n_clips]
    if not files:
        print("No full-length validation clips found.", file=sys.stderr)
        return 2
    shifts = [float(s) for s in args.shifts.split(",")]
    print(f"{len(files)} held-out clips, checkpoint {ckpt}, shifts {shifts}")

    acc = {(name, s): None for name in names for s in shifts}
    for i in tqdm(range(0, len(files), args.batch_size), desc="batches"):
        loaded = [np.load(f) for f in files[i : i + args.batch_size]]
        # the first training-clip's worth of each stored segment
        n_frames = 1 + CLIP_LEN // cfg.hop_length
        clips = np.stack([d["audio"][:CLIP_LEN] for d in loaded]).astype(np.float32)
        f0_src = np.stack([d["f0"][:n_frames] for d in loaded])
        conf_src = np.stack([d["conf"][:n_frames] for d in loaded])
        mel_src = compute_mel_spectrogram(clips)
        for name in names:
            for s in shifts:
                part = score_batch(clips, f0_src, conf_src, mel_src, s, systems[name])
                if acc[(name, s)] is None:
                    acc[(name, s)] = part
                else:
                    for k, v in part.items():
                        acc[(name, s)][k].extend(v)

    results = []
    for s in shifts + ["all"]:
        for name in names:
            keys = [(name, x) for x in shifts] if s == "all" else [(name, s)]
            merged = {k: sum((acc[key][k] for key in keys), []) for k in acc[keys[0]]}
            row = summarise(merged)
            if name == "vocoder only":
                for k in ("f0_median_cents", "f0_gross_pct", "voiced_kept_pct"):
                    row[k] = float("nan")
            results.append({"shift": s, "system": name, **row})

    cols = [
        ("f0_median_cents", "F0 err (cents)"),
        ("f0_gross_pct", f"F0 >{GROSS_CENTS:g}c (%)"),
        ("voiced_kept_pct", "voiced kept (%)"),
        ("envelope_dist", "envelope dist"),
        ("roundtrip_mel_l1", "round-trip mel L1"),
        ("roundtrip_mcd_db", "round-trip MCD (dB)"),
    ]
    print("\n| shift | system | " + " | ".join(label for _, label in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for r in results:
        shift = r["shift"] if r["shift"] == "all" else f"{r['shift']:+g}"
        cells = " | ".join(f"{r[k]:.3f}" for k, _ in cols)
        print(f"| {shift} | {r['system']} | {cells} |")

    args.out.write_text(
        json.dumps(
            {"ckpt": str(ckpt), "n_clips": len(files), "results": results}, indent=2
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
