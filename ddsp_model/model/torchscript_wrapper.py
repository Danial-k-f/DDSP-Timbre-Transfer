import torch
import torch.nn as nn

class TorchScriptWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        out = self.model(batch)
        return (
            out["harmonic"],
            out["noise"],
            out["audio_synth"],
            out["audio_reverb"] if "audio_reverb" in out else torch.zeros_like(out["audio_synth"]),
            out["a"], out["c"], out["H"], out["f0"], out["loudness"], out["z"]
        )
