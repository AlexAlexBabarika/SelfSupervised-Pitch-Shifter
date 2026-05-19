import numpy as np


def overlap_add(clips: np.ndarray, hop: int, clip_len: int) -> np.ndarray:
    assert clips.ndim == 2, f"clips must be [N, T], got {clips.shape}"
    assert clips.shape[1] == clip_len, (
        f"clip width {clips.shape[1]} != clip_len {clip_len}"
    )
    N = clips.shape[0]
    if N == 1:
        return clips[0].astype(np.float32, copy=True)

    overlap = clip_len - hop
    assert 0 < overlap <= clip_len, f"hop {hop} must give overlap in (0, clip_len]"

    out_len = hop * (N - 1) + clip_len
    out = np.zeros(out_len, dtype=np.float32)

    phase = np.pi * (np.arange(overlap, dtype=np.float64) + 0.5) / overlap
    ramp_up = (0.5 * (1.0 - np.cos(phase))).astype(np.float32)
    ramp_down = (1.0 - ramp_up).astype(np.float32)

    def _windowed(i: int) -> np.ndarray:
        w = np.ones(clip_len, dtype=np.float32)
        if i > 0:
            w[:overlap] = ramp_up
        if i < N - 1:
            w[-overlap:] = ramp_down
        return clips[i].astype(np.float32, copy=False) * w

    for i in range(N):
        start = i * hop
        out[start : start + clip_len] += _windowed(i)

    return out
