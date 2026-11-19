from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    target_lufs: int = -23
    target_sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    win_length: int = 1024
    clip_seconds: float = 2.56
    segment_seconds: float = 5.12
    crepe_model_capacity: str = "full"
    voiced_threshold: float = 0.3
    fmin: int = 0
    fmax: int = 8000


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    subdir: str
    length_hr: int
    parquet_glob: str
    group_minutes: float = 5.0


@dataclass
class DataConfig:
    cache_dir: str = "./cache"
    val_split: float = 0.02

    datasets: dict[str, DatasetSpec] = field(
        default_factory=lambda: {
            "nsynth": DatasetSpec(
                hf_name="jg583/NSynth",
                subdir="nsynth",
                length_hr=2,
                parquet_glob="data/train/*.parquet",
            ),
            "vctk": DatasetSpec(
                hf_name="sanchit-gandhi/vctk",
                subdir="vctk",
                length_hr=10,
                parquet_glob="data/train-*.parquet",
            ),
            "opensinger_female": DatasetSpec(
                hf_name="CodecSR/opensinger_female",
                subdir="opensinger_fem",
                length_hr=3,
                parquet_glob="data/original-*.parquet",
                group_minutes=90,
            ),
            "opensinger_male": DatasetSpec(
                hf_name="CodecSR/opensinger_male",
                subdir="opensinger_male",
                length_hr=3,
                parquet_glob="data/original-*.parquet",
                group_minutes=90,
            ),
        }
    )
    datasets_to_load: list[str] = field(
        default_factory=lambda: [
            "vctk",
            "opensinger_female",
            "opensinger_male",
        ]
    )

    def __post_init__(self):
        assert self.val_split < 1
        unknown = set(self.datasets_to_load) - self.datasets.keys()
        assert not unknown, f"unknown datasets in datasets_to_load: {unknown}"


@dataclass
class TrainConfig:
    perturb_st: int = 6
    batch_size: int = 32
    num_workers: int = 6
    warmup_steps: int = 1000
    max_steps: int = 60000
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    betas = (0.8, 0.99)
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    cond_dropout: float = 0.0
    skip_dropout: float = 0.0
    log_every: int = 100
    val_every: int = 2000
    save_every: int = 2000
    keep_last: int = 5
    ckpt_dir: str = "./checkpoints"
    tb_dir: str = "./runs"
    # Waveform loss through the frozen vocoder; w_wave = 0 turns it off.
    w_wave: float = 0.5
    wave_batch: int = 8
    wave_frames: int = 64
    vocoder_repo: str = "nvidia/bigvgan_base_22khz_80band"
    use_wandb: bool = False
    wandb_project: str = "pitch-shifter"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None


@dataclass
class ModelConfig:
    base_channels: int = 64
    channel_mults = [1, 2, 4, 6]  # 64, 128, 256, 384
    shift_emb_dim: int = 64
    f0_emb_dim: int = 64
    attention_in_bottleneck: bool = True
