# pitch-shifter

A neural pitch shifter for vocals and music. Instead of warping audio with a
classic phase vocoder, it predicts the *target* mel-spectrogram of a pitch-shifted
signal with a conditional U-Net and resynthesizes audio with a HiFi-GAN vocoder.
For full songs it separates stems with Demucs, shifts only the vocals, and remixes.

## How it works

```
song.wav
  └─ Demucs (htdemucs) ── vocals ──► [resample 22.05 kHz]
                       │                     │
                       │              log-mel + F0 (CREPE)
                       │                     │
                       │              PitchUNet (mel → mel)
                       │                     │
                       │              HiFi-GAN (mel → audio)
                       │                     │
                       └─ drums/bass/other ──┴──► remix ──► shifted.wav
```

The core model, **`PitchUNet`** (`model.py`), is a 2-D U-Net over log-mel
spectrograms. It is conditioned on:

- **Shift amount** (semitones), embedded and injected through FiLM layers in every
  residual block. A `cond_mask` distinguishes "no conditioning" from "shift = 0",
  enabling classifier-free-style conditioning dropout during training.
- **F0 contour** (`log2(f0+1)` + voiced mask), extracted with
  [`torchcrepe`](https://github.com/maxrmorrison/torchcrepe). F0 features are
  concatenated at the input and also drive **gated skip connections**, so the
  decoder leans on the pitch track when reconstructing harmonics.

A self-attention block sits in the bottleneck. The output head is zero-initialized
so early training predicts ≈0 (i.e. the loss starts as `|target|`).

Training (`train.py`) uses an L1 + multi-resolution mel loss (`losses.py`), AdamW
with cosine decay and warmup, EMA weights, gradient clipping, and bf16 autocast on
CUDA. The input mel is produced by a *real* audio-domain `librosa.pitch_shift`, so
the network learns to clean up the artifact profile of a phase-vocoder shift rather
than a trivial mel-bin warp.

## Layout

| Path | Purpose |
|------|---------|
| `config.py` | All hyperparameters (`AudioConfig`, `DataConfig`, `TrainConfig`, `ModelConfig`) |
| `Download_HF_Datasets.py` | Pull dataset parquet shards from the Hugging Face Hub |
| `Process_HF_Datasets.py` | Stream parquet → preprocessed `.npz` clips + index files |
| `audio_preprocessing.py` | Loudness norm, segmentation, mel + CREPE F0 extraction |
| `data.py` | `PitchDataset` / dataloader; applies the train-time pitch perturbation |
| `model.py` | `PitchUNet` and conditioning modules |
| `losses.py` | L1 + multi-resolution mel loss |
| `train.py` | Training loop, EMA, checkpointing, TensorBoard / W&B logging |
| `inference.py` | CLI entrypoint: separate → shift → remix |
| `inference/` | Demucs separation, HiFi-GAN vocoder, overlap-add stitching, pipeline |

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

To download datasets you need a Hugging Face token:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=...
```

## Inference

Shift a song down two semitones (separates stems, shifts vocals, remixes):

```bash
uv run inference.py path/to/song.wav -s -2
```

Useful flags:

- `-s, --semitones` — shift amount; positive = up, negative = down (required)
- `-o, --output` — output path (default: `<input>_shifted_<+N>st.wav` next to input)
- `--no-separate` — skip Demucs and shift the whole mix as a single mono signal
- `--ckpt` — PitchUNet checkpoint (default: `./final.pt`, else latest `./checkpoints/step_*.pt`)
- `--no-ema` — use raw weights instead of the EMA shadow
- `--vocoder-repo` — HF repo hosting the HiFi-GAN UNIVERSAL_V1 generator
  (default: `alexalexbabarika/hifigan-universal-v1`)
- `--device` — `auto` | `cuda` | `mps` | `cpu`
- `-v, --verbose` — progress output

When separating, the individual stems are also written to a `*_stems/` directory
next to the output.

## Training

```bash
# 1. download dataset shards into ./datasets
uv run Download_HF_Datasets.py

# 2. preprocess into ./cache/<dataset>/*.npz (+ index json)
uv run Process_HF_Datasets.py

# 3. train
uv run train.py
```

Datasets, the per-dataset hour budget, and the train/val split are configured in
`DataConfig`. Out of the box it mixes NSynth (vocal-range notes only), VCTK
speech, and OpenSinger (male/female). Checkpoints land in `./checkpoints/`
(pruned to the last few), with `final.pt` written at the end of training; logs go
to `./runs/` (TensorBoard) and optionally Weights & Biases.

## Notes

- All preprocessing and training run at 22.05 kHz mono with 80-bin mels
  (`n_fft=1024`, `hop=256`); Demucs operates at its native 44.1 kHz stereo.
- The bundled HiFi-GAN code (`inference/hifigan/`) is the standard UNIVERSAL_V1
  generator; weights are fetched from the Hub at inference time.
