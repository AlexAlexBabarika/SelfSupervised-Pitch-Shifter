import torch
import torchaudio.functional as AF
from demucs.apply import apply_model
from demucs.pretrained import get_model


_MODEL = None
_MODEL_DEVICE: str | None = None


def get_separator(device: torch.device):
    global _MODEL, _MODEL_DEVICE
    dev_str = str(device)
    if _MODEL is None:
        _MODEL = get_model("htdemucs")
        _MODEL.eval()
    if _MODEL_DEVICE != dev_str:
        _MODEL.to(device)
        _MODEL_DEVICE = dev_str
    return _MODEL


def separate_stems(
    audio: torch.Tensor, sr: int, device: torch.device
) -> dict[str, torch.Tensor]:
    # audio: [2, samples] stereo at any sample rate. Returns {"vocals",
    # "drums", "bass", "other"} as [2, S] tensors on CPU at the model's
    # native sample rate (44.1 kHz for htdemucs).
    model = get_separator(device)
    if sr != model.samplerate:
        audio = AF.resample(audio, sr, model.samplerate)
    audio = audio.to(device)

    ref = audio.mean(0)
    audio = (audio - ref.mean()) / ref.std()

    with torch.inference_mode():
        sources = apply_model(
            model,
            audio[None],
            split=True,
            overlap=0.25,
            progress=False,
            device=device,
        )  # [1, n_sources, channels, samples]

    sources = sources[0] * ref.std() + ref.mean()  # [n_sources, channels, samples]
    return {name: sources[i].detach().cpu() for i, name in enumerate(model.sources)}
