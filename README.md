# pitch-shifter

A pitch shifter for the vocals of a song. It separates stems with Demucs, shifts
only the vocals, and remixes. There are two shifting engines:

- **`rubberband`** (default): [Rubber Band](https://breakfastquay.com/rubberband/)
  with control over how far the formants follow the pitch. In `evaluate.py` and in
  listening comparisons it beat WORLD, which sounded metallic with or without the
  vocoder; the U-Net had not been retrained when that comparison was made.
- **`model`** (experimental): a formant-preserving WORLD shift moves the pitch, a
  conditional U-Net predicts the clean mel-spectrogram of the shifted signal, and a
  BigVGAN vocoder resynthesizes the audio. It needs a trained checkpoint.

## How the model engine works

```
song.wav
  └─ Demucs (htdemucs) ── vocals ──► mid ──► [resample 22.05 kHz]
                       │          │                 │
                       │          │          F0 (CREPE)
                       │          │                 │
                       │          │          WORLD shift ──► log-mel
                       │          │                 │
                       │          │          PitchUNet (mel → mel)
                       │          │                 │
                       │          │          BigVGAN (mel → audio)
                       │          │                 │
                       │          │          + band above 8 kHz from a
                       │          │            44.1 kHz WORLD shift
                       │          │                 │
                       │          └─ side ─► phase-vocoder shift
                       │                            │
                       └─ drums/bass/other ─────────┴──► remix ──► shifted.wav
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
- **Harmonic template**: the F0 track rendered as a picture of where its harmonics
  fall on the mel axis, concatenated at the input.

A self-attention block sits in the bottleneck. The network predicts a residual on
its input mel and the output head is zero-initialized, so training starts from the
identity (i.e. from the quality of the WORLD shift).

Training (`train.py`) uses an L1 + multi-resolution + difference loss on the mel and
a multi-resolution STFT loss on audio rendered through the frozen vocoder
(`losses.py`; set `w_wave = 0` to turn the latter off), AdamW
with cosine decay and warmup, EMA weights, gradient clipping, and bf16 autocast on
CUDA. The input mel comes from a WORLD round trip (shift by `+s`, then back by
`-s`) of the original clip, so the network learns to remove the artifacts of a
WORLD shift while the target stays the untouched original. Inference applies a
single WORLD shift and hands the result to the network. WORLD models one voice,
so backing vocals and heavy reverb in the input degrade the result.

At inference F0 and WORLD run once over the whole signal, the U-Net runs over
overlapping windows whose predicted mels are cross-faded, and the vocoder makes a
single pass over the stitched mel. For stereo vocals the mid channel goes through
the model and the side channel gets a plain phase-vocoder shift.

## Layout

| Path | Purpose |
|------|---------|
| `config.py` | All hyperparameters (`AudioConfig`, `DataConfig`, `TrainConfig`, `ModelConfig`) |
| `Download_HF_Datasets.py` | Pull dataset parquet shards from the Hugging Face Hub |
| `Process_HF_Datasets.py` | Stream parquet → preprocessed `.npz` clips + index files |
| `audio_preprocessing.py` | Loudness norm, segmentation, mel + CREPE F0 extraction, WORLD shift |
| `data.py` | `PitchDataset` / dataloader; applies the train-time WORLD round trip |
| `model.py` | `PitchUNet` and conditioning modules |
| `losses.py` | Mel losses and the waveform loss through the frozen vocoder |
| `train.py` | Training loop, EMA, checkpointing, TensorBoard / W&B logging |
| `inference.py` | CLI entrypoint: separate → shift → remix |
| `inference/` | Demucs separation, vocoders (BigVGAN, HiFi-GAN), stitching, pipeline |

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

The default engine needs the `rubberband` command-line tool
(`brew install rubberband` on macOS, `apt install rubberband-cli` on Debian/Ubuntu).

Useful flags:

- `-s, --semitones` — shift amount; positive = up, negative = down (required)
- `-o, --output` — output path (default: `<input>_shifted_<+N>st.wav` next to input)
- `--engine` — `rubberband` (default) or `model`
- `--formant-shift` — semitones the formants move, `rubberband` engine only. The
  default keeps them fixed when shifting down and lets them follow a quarter of the
  way when shifting up, as a real voice does; `0` is full formant preservation
- `--no-separate` — skip Demucs and shift the input as it is; meant for solo vocal
  recordings (the `model` engine only handles a single voice)

Flags for the `model` engine:

- `--ckpt` — PitchUNet checkpoint (default: `./best.pt`, else `./final.pt`, else latest
  `./checkpoints/step_*.pt`)
- `--no-ema` — use raw weights instead of the EMA shadow
- `--vocoder-repo` — HF repo of the vocoder; BigVGAN and HiFi-GAN repos both work
  (default: `TrainConfig.vocoder_repo`, the one the model was trained against)
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
`DataConfig`. Validation holds out whole speakers, singers and instruments, so
the split is at least `val_split` of each dataset rather than exactly that. Out of the box it mixes VCTK speech and OpenSinger (male/female);
NSynth is still defined but no longer loaded. Checkpoints land in `./checkpoints/`
(pruned to the last few), with `best.pt` written whenever validation improves and
`final.pt` at the end of training; logs go
to `./runs/` (TensorBoard) and optionally Weights & Biases.

## Evaluation

```bash
uv run evaluate.py            # uses ./best.pt and the held-out validation clips
```

Scores the full system against `librosa`, Rubber Band (if the `rubberband` CLI is
installed), PSOLA (Praat), WORLD alone, WORLD + vocoder (no U-Net) and the vocoder
alone, per shift size. A single shift has no ground truth, so it is
scored on pitch accuracy (CREPE F0 against the source F0 scaled by the shift), on
how many voiced frames survive, and on how far the spectral envelope moved from the
source. A round trip (`+s` then `-s`) is compared with the original clip by log-mel
L1 and mel-cepstral distortion. Results are printed as a table and written to
`eval_results.json`.

## Notes

- All preprocessing and training run at 22.05 kHz mono with 80-bin mels
  (`n_fft=1024`, `hop=256`, up to 8 kHz); Demucs operates at its native 44.1 kHz
  stereo, and inference output is 44.1 kHz.
- The cache holds segments of up to 5.12 s; each training step crops a 2.56 s clip
  at a random offset.
- The default vocoder is BigVGAN base (22 kHz, 80 bands); its code is vendored in
  `inference/bigvgan/` (MIT, NVIDIA). The HiFi-GAN UNIVERSAL_V1 generator in
  `inference/hifigan/` is still supported. Weights are fetched from the Hub.
