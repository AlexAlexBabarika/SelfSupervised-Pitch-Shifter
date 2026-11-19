#!/usr/bin/env bash
# Train PitchUNet on a rented GPU box (written for Vast.ai, works on any
# Ubuntu machine with an NVIDIA driver). Run from the repository root:
#
#   HF_TOKEN=hf_... bash scripts/vast.sh setup   # system packages, uv, Python deps
#   bash scripts/vast.sh data                    # download + preprocess + sanity check
#   bash scripts/vast.sh smoke                   # 60 steps end to end, a few minutes
#   bash scripts/vast.sh train                   # the real run; resumes if interrupted
#   bash scripts/vast.sh eval                    # evaluate.py on best.pt
#
# Run `train` inside tmux so it survives a dropped SSH connection.
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
WORKERS="${WORKERS:-$(( NPROC > 2 ? NPROC - 2 : 1 ))}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

setup() {
    $SUDO apt-get update
    # build-essential: pyworld compiles from source. rubberband-cli: only for
    # the Rubber Band row of evaluate.py.
    $SUDO apt-get install -y build-essential git tmux curl
    $SUDO apt-get install -y rubberband-cli || echo "rubberband-cli not available; evaluate.py will skip that row"
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv sync
    if [ -n "${HF_TOKEN:-}" ] && [ ! -f .env ]; then
        (umask 077; printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env)
    fi
    uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'; print('GPU:', torch.cuda.get_device_name(0))"
}

data() {
    uv run Download_HF_Datasets.py
    uv run Process_HF_Datasets.py
    uv run python - <<'PY'
import json, random
from pathlib import Path
import numpy as np
from config import AudioConfig, DataConfig

cfg, data = AudioConfig(), DataConfig()
random.seed(0)
for key in data.datasets_to_load:
    spec = data.datasets[key]
    files = json.loads((Path(data.cache_dir) / f"{spec.subdir}_index.json").read_text())
    groups = {Path(f).name.split("__")[0] for f in files}
    voiced, f0 = [], []
    for f in random.sample(files, min(200, len(files))):
        d = np.load(f)
        v = d["conf"] > cfg.voiced_threshold
        voiced.append(v.mean())
        f0.extend(d["f0"][v].tolist())
    hours = sum(1 for _ in files) * cfg.segment_seconds / 3600
    med = float(np.median(f0)) if f0 else float("nan")
    print(f"{key}: {len(files)} segments (<= {hours:.1f} h), {len(groups)} voices, "
          f"{100 * np.mean(voiced):.0f}% frames voiced, median F0 {med:.0f} Hz")
    assert len(groups) > 1, f"{key}: need at least 2 voices for a validation split"
    assert 0.15 < np.mean(voiced) < 0.98, f"{key}: voiced share looks wrong"
    assert 70 < med < 700, f"{key}: median F0 looks wrong (the old CREPE bug gave ~2000 Hz)"
print("cache looks sane")
PY
}

free_space() {
    echo "free disk space here: $(df -h . | awk 'NR==2 {print $4}')"
}

smoke() {
    # Kept next to the repository: /tmp sits on the container's small root
    # disk on Vast.ai, and each checkpoint is about 300 MB.
    free_space
    rm -rf .smoke
    uv run train.py --max-steps 60 --val-every 30 --save-every 20 \
        --num-workers "$WORKERS" --out-dir .smoke
    rm -rf .smoke
    echo "smoke test passed"
}

train() {
    free_space
    uv run train.py --num-workers "$WORKERS" 2>&1 | tee -a train.log
}

evaluate() {
    uv run evaluate.py 2>&1 | tee eval.log
}

case "${1:-}" in
    setup) setup ;;
    data) data ;;
    smoke) smoke ;;
    train) train ;;
    eval) evaluate ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
