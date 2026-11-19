import json
import random
import numpy as np
import torch
import torchaudio
from typing import List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import DataConfig, TrainConfig, AudioConfig
from audio_preprocessing import CLIP_LEN, f0_to_feature, world_pitch_shift

data_cfg = DataConfig()
audio_cfg = AudioConfig()
train_cfg = TrainConfig()

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=audio_cfg.target_sr,
    n_fft=audio_cfg.n_fft,
    hop_length=audio_cfg.hop_length,
    win_length=audio_cfg.win_length,
    n_mels=audio_cfg.n_mels,
    f_min=audio_cfg.fmin,
    f_max=audio_cfg.fmax,
    power=1.0,
    center=True,
    norm="slaney",
    mel_scale="slaney",
)


def _audio_to_log_mel(audio: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(audio).float().unsqueeze(0)
    m = _mel(t).squeeze(0)
    return torch.log(m.clamp(min=1e-5))


def _split_by_group(files: List[str], val_split: float):
    # Clip names are "<group>__<rest>.npz", group being the speaker, singer or
    # instrument. Whole groups go to val so no voice is on both sides.
    groups: dict[str, List[str]] = {}
    for f in files:
        name = Path(f).name
        assert "__" in name, f"{name} has no group prefix; rerun Process_HF_Datasets.py"
        groups.setdefault(name.split("__")[0], []).append(f)
    assert len(groups) > 1, f"need at least 2 groups to split, got {list(groups)}"
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    val: List[str] = []
    while len(val) < len(files) * val_split and len(keys) > 1:
        val.extend(groups[keys.pop()])
    return [f for k in keys for f in groups[k]], val


class PitchDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_keys: List[str] = data_cfg.datasets_to_load,
    ):
        self.files = []
        # 1 / (segments in the file's dataset): sampling with these weights
        # draws every dataset equally often, so the larger speech set does
        # not crowd out the singing.
        self.weights: List[float] = []
        for key in dataset_keys:
            spec = data_cfg.datasets[key]
            with open(Path(data_cfg.cache_dir) / f"{spec.subdir}_index.json") as f:
                train, val = _split_by_group(json.load(f), data_cfg.val_split)
            part = train if split == "train" else val
            self.files.extend(part)
            self.weights.extend([1.0 / max(1, len(part))] * len(part))
        self.perturb_st = train_cfg.perturb_st
        self.is_train = split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as d:
            audio = d["audio"].astype(np.float32, copy=True)
            f0_raw = d["f0"].copy()  # Hz, aligned to mel frames
            conf = d["conf"].copy()

        # Crop one training clip out of the stored segment, at a random frame
        # offset when training. Frame i is centred on sample i * hop.
        hop = audio_cfg.hop_length
        n_frames = 1 + CLIP_LEN // hop
        max_k = max(0, (len(audio) - CLIP_LEN) // hop)
        k = random.randint(0, max_k) if self.is_train else 0
        audio = audio[k * hop : k * hop + CLIP_LEN]
        real_len = len(audio)
        audio = np.pad(audio, (0, CLIP_LEN - real_len))
        f0_raw = f0_raw[k : k + n_frames]
        conf = conf[k : k + n_frames]
        mel_tgt = _audio_to_log_mel(audio)

        if self.is_train:
            semis = random.uniform(-self.perturb_st, self.perturb_st)
        else:
            rng = random.Random(index)
            semis = rng.uniform(-self.perturb_st, self.perturb_st)
            if abs(semis) < 0.5:
                semis = 0.5 if semis >= 0 else -0.5

        # WORLD round trip: shift by +semis, then back by -semis. mel_in sits at
        # the original pitch and carries the artifact profile of a WORLD shift,
        # which is what inference feeds the model after its single shift.
        if abs(semis) > 1e-4:
            ratio = 2.0 ** (semis / 12.0)
            shifted = world_pitch_shift(audio, f0_raw, conf, semis)
            audio_in = world_pitch_shift(shifted, f0_raw * ratio, conf, -semis)
        else:
            audio_in = audio
        mel_in = _audio_to_log_mel(audio_in)

        f0_feat = f0_to_feature(
            torch.from_numpy(f0_raw), torch.from_numpy(conf), fmax=audio_cfg.fmax
        )  # [2, T]

        # 1 on frames that hold real audio, 0 on the zero-padded tail
        mask = torch.zeros(1, 1, mel_tgt.shape[-1])
        mask[..., : 1 + real_len // hop] = 1.0

        return {
            "mel_in": mel_in.unsqueeze(0),  # [1, 80, T]
            "mel_tgt": mel_tgt.unsqueeze(0),  # [1, 80, T]
            "f0": f0_feat,  # [2, T]
            "mask": mask,  # [1, 1, T]
            "audio": torch.from_numpy(audio),  # [CLIP_LEN], clean
            "shift": torch.tensor(-semis, dtype=torch.float32),
        }

    @staticmethod
    def make_loader(split="train", batch_size=None, num_workers=None):
        ds = PitchDataset(split)
        is_train = split == "train"
        if num_workers is None:
            num_workers = train_cfg.num_workers
        nw = num_workers if is_train else min(4, num_workers)
        sampler = None
        if is_train:
            sampler = WeightedRandomSampler(ds.weights, len(ds), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size or train_cfg.batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=(nw > 0),
            multiprocessing_context="spawn" if nw > 0 else None,
        )
