import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# How far the formants follow an upward shift. A real voice's formants rise a
# little with pitch; downwards they stay put, or the singer changes size.
FORMANT_FOLLOW_UP = 0.25


def default_formant_shift(semis: float) -> float:
    return FORMANT_FOLLOW_UP * semis if semis > 0 else 0.0


def _run(audio: np.ndarray, sr: int, semis: float, formant: bool) -> np.ndarray:
    # audio: [samples] or [channels, samples]. Rubber Band R3 via its CLI.
    if abs(semis) < 1e-6:
        return audio
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        sf.write(src, audio.T, sr, subtype="FLOAT")
        cmd = ["rubberband", "-q", "--fine", f"--pitch={semis}", src, dst]
        if formant:
            cmd.insert(3, "--formant")
        if audio.ndim == 2 and audio.shape[0] == 2:
            # Without this the channels drift out of phase: the centre loses
            # level and the sides nearly double.
            cmd.insert(3, "--centre-focus")
        subprocess.run(cmd, check=True, capture_output=True)
        out, _ = sf.read(dst, dtype="float32")
    out = out.T
    n = audio.shape[-1]
    if out.shape[-1] < n:
        pad = [(0, 0)] * (out.ndim - 1) + [(0, n - out.shape[-1])]
        out = np.pad(out, pad)
    return np.ascontiguousarray(out[..., :n])


def rubberband_shift(
    audio: np.ndarray, sr: int, semis: float, formant_semis: float = 0.0
) -> np.ndarray:
    # Moves the pitch by `semis` and the formants by `formant_semis`: one pass
    # moves both together, a second moves the pitch alone. Stereo input
    # keeps its image and its mono compatibility; the level is kept too.
    if shutil.which("rubberband") is None:
        raise RuntimeError(
            "The `rubberband` command-line tool is required "
            "(macOS: `brew install rubberband`; Debian/Ubuntu: `apt install rubberband-cli`)."
        )
    audio = audio.astype(np.float32, copy=False)
    both = _run(audio, sr, formant_semis, formant=False)
    out = _run(both, sr, semis - formant_semis, formant=True)
    # Formant preservation changes the level by a couple of dB; put it back
    # so the vocal keeps its place in the mix.
    rms_in, rms_out = np.sqrt((audio**2).mean()), np.sqrt((out**2).mean())
    if rms_out > 1e-8:
        out = out * (rms_in / rms_out)
    return out.astype(np.float32)
