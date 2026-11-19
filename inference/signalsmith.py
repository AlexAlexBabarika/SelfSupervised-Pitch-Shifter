import numpy as np
import python_stretch


def signalsmith_shift(
    audio: np.ndarray, sr: int, semis: float, tonality_hz: float = 8000.0
) -> np.ndarray:
    # audio: [samples] or [channels, samples]. Signalsmith Stretch, for full
    # music: it has no formant control, so vocals go through Rubber Band.
    # Above tonality_hz the spectrum is left in place, which keeps cymbals
    # and air from turning metallic.
    if abs(semis) < 1e-6:
        return audio
    x = np.ascontiguousarray(np.atleast_2d(audio), dtype=np.float32)
    stretch = python_stretch.Signalsmith.Stretch()
    stretch.preset(x.shape[0], sr)
    stretch.setTransposeSemitones(semis, tonality_hz / sr)
    out = stretch.process(x)
    return out[0] if audio.ndim == 1 else out
