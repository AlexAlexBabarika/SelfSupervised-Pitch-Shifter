import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import librosa
import numpy as np
import torch
import torchaudio

from config import TrainConfig
from inference.checkpoint import find_ckpt, load_unet


DEMUCS_SR = 44100


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _peak_normalize(audio: np.ndarray, headroom: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) + 1e-9
    if peak > headroom:
        audio = audio * (headroom / peak)
    return audio.astype(np.float32, copy=False)


def _default_output(input_path: Path, semis: float) -> Path:
    sign = "+" if semis >= 0 else "-"
    mag = f"{abs(semis):g}"
    return input_path.with_name(f"{input_path.stem}_shifted_{sign}{mag}st.wav")


def _match_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[-1] >= length:
        return x[..., :length]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, length - x.shape[-1])]
    return np.pad(x, pad)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    if audio.ndim == 1:
        tensor = torch.from_numpy(audio).unsqueeze(0)
    else:
        tensor = torch.from_numpy(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), tensor, sr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pitch-shift a song through PitchUNet + HiFi-GAN."
    )
    p.add_argument("input", type=Path, help="Input audio file (wav/mp3/flac/...).")
    p.add_argument(
        "-s",
        "--semitones",
        type=float,
        required=True,
        help="Semitone shift. Positive = up, negative = down.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output wav path. Default: <input_stem>_shifted_<+N>st.wav next to input.",
    )
    p.add_argument(
        "--no-separate",
        action="store_true",
        help="Skip Demucs. For solo vocal recordings only: the shifter models "
        "a single voice and will not work on a full mix.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="PitchUNet checkpoint. Default: ./best.pt, else ./final.pt, "
        "else latest step_*.pt under ./checkpoints/.",
    )
    p.add_argument(
        "--no-ema",
        action="store_true",
        help="Load raw model weights instead of the EMA shadow.",
    )
    p.add_argument(
        "--vocoder-repo",
        default=TrainConfig().vocoder_repo,
        help="HF repo id of the vocoder (HiFi-GAN or BigVGAN). Should be the "
        "one the model was trained against.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Clip batch size for the UNet+vocoder loop.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    max_shift = TrainConfig().perturb_st
    if abs(args.semitones) > max_shift:
        print(
            f"Warning: the model was trained on shifts up to ±{max_shift} semitones; "
            f"{args.semitones:+g} is outside that range.",
            file=sys.stderr,
        )

    device = _resolve_device(args.device)

    ckpt = args.ckpt or find_ckpt(Path("./checkpoints"))
    if ckpt is None:
        print(
            "No checkpoint found in ./checkpoints/ and --ckpt not given.",
            file=sys.stderr,
        )
        return 2
    if args.verbose:
        print(f"Loading PitchUNet from {ckpt}")
    unet = load_unet(ckpt, device, use_ema=not args.no_ema)

    # Local imports keep cold startup snappy if -h is the only ask.
    from inference.pipeline import shift_audio_through_model
    from inference.vocoder import load_vocoder

    if args.verbose:
        print(f"Loading vocoder from {args.vocoder_repo}")
    vocoder = load_vocoder(args.vocoder_repo, device)

    default_name = _default_output(args.input, args.semitones).name
    if args.output is None:
        output = _default_output(args.input, args.semitones)
    elif args.output.is_dir() or str(args.output).endswith("/"):
        output = args.output / default_name
    else:
        output = args.output

    if args.no_separate:
        if args.verbose:
            print(f"Loading {args.input} (mono @ {DEMUCS_SR} Hz)")
        audio, _ = librosa.load(args.input, sr=DEMUCS_SR, mono=True)
        if args.verbose:
            print(f"Shifting by {args.semitones:+g} semitones")
        shifted = shift_audio_through_model(
            audio,
            DEMUCS_SR,
            args.semitones,
            unet,
            vocoder,
            device,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        shifted = _peak_normalize(shifted)
        _write_wav(output, shifted, DEMUCS_SR)
        if args.verbose:
            print(f"Wrote {output} @ {DEMUCS_SR} Hz")
        return 0

    from inference.separation import separate_stems

    if args.verbose:
        print(f"Loading {args.input} (stereo @ {DEMUCS_SR} Hz)")
    audio, _ = librosa.load(args.input, sr=DEMUCS_SR, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    audio_t = torch.from_numpy(audio.astype(np.float32))

    if args.verbose:
        print("Running Demucs separation")
    stems = separate_stems(audio_t, DEMUCS_SR, device)

    stems_dir = output.with_name(f"{output.stem}_stems")
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in stems.items():
        _write_wav(stems_dir / f"{name}.wav", tensor.numpy(), DEMUCS_SR)
    if args.verbose:
        print(f"Saved stems to {stems_dir}")

    # Mid/side: the mid (what both channels share, i.e. the voice) goes
    # through the model; the side (reverb, doubles) is not a single voice, so
    # it gets a plain phase-vocoder shift and the stereo image is rebuilt.
    vocals = stems["vocals"].numpy()
    mid = vocals.mean(axis=0)
    side = (vocals[0] - vocals[1]) / 2.0
    if args.verbose:
        print(f"Shifting vocals by {args.semitones:+g} semitones")
    mid = shift_audio_through_model(
        mid,
        DEMUCS_SR,
        args.semitones,
        unet,
        vocoder,
        device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    if float(np.abs(side).max()) > 1e-6:
        side = librosa.effects.pitch_shift(side, sr=DEMUCS_SR, n_steps=args.semitones)
        side = _match_length(side, mid.shape[-1])
    shifted_44k = np.stack([mid + side, mid - side], axis=0)

    target_len = max(shifted_44k.shape[-1], int(stems["drums"].shape[-1]))
    mix = _match_length(shifted_44k, target_len).copy()
    for name in ("drums", "bass", "other"):
        s = stems[name].numpy()
        mix = mix + _match_length(s, target_len)

    mix = _peak_normalize(mix)
    _write_wav(output, mix, DEMUCS_SR)
    if args.verbose:
        print(f"Wrote {output} @ {DEMUCS_SR} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
