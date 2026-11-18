import re
from pathlib import Path

import torch

from model import PitchUNet

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def find_ckpt(ckpt_dir: Path) -> Path | None:
    for name in ("best.pt", "final.pt"):
        if Path(name).exists():
            return Path(name)
    if not ckpt_dir.exists():
        return None
    candidates = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = _STEP_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_unet(ckpt_path: Path, device: torch.device, use_ema: bool) -> PitchUNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PitchUNet().to(device)
    model.load_state_dict(ckpt["model"])
    if use_ema:
        shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in shadow:
                    p.copy_(shadow[n].to(device))
    model.eval()
    return model
