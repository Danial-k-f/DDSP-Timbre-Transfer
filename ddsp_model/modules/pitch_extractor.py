import torch
import librosa

class YINPitchExtractor:
    def __init__(self, sr=44100, frame_length=2048, hop_length=220, fmin=50.0, fmax=2000.0):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, audio: torch.Tensor):
        """
        audio: (B, T) torch tensor
        return: (B, frames) f0 values
        """
        audio_np = audio.squeeze(0).cpu().numpy()
        f0 = librosa.yin(
            audio_np,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        return torch.tensor(f0, dtype=torch.float32).unsqueeze(0)
