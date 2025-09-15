import torch
import torch.nn as nn
import numpy as np


class HarmonicOscillator(nn.Module):
    def __init__(self, sr=48000, frame_resolution=0.005, attenuate_gain=0.02, device="cpu"):
        super().__init__()
        self.sr = sr
        self.frame_resolution = frame_resolution
        self.hop_length = int(sr * frame_resolution)   # ✅ همیشه درست محاسبه میشه
        self.attenuate_gain = attenuate_gain
        self.device = device

        # Upsample از frame-rate → audio-rate
        self.framerate_to_audiorate = nn.Upsample(
            scale_factor=self.hop_length, mode="linear", align_corners=False
        )

    def forward(self, f0, a, c):
        """
        Args:
            f0: [B, T] fundamental frequency
            a:  [B, T] amplitude
            c:  [B, n_harmonics, T] harmonic distribution
        Returns:
            audio: [B, samples]
        """
        B, T = f0.shape
        num_osc = c.shape[1]  # n_harmonics

        # Harmonic frequencies
        partials = torch.arange(1, num_osc + 1, dtype=torch.float32, device=f0.device).view(1, -1, 1)  # [1, n_harmonics, 1]
        f0_bank = f0.unsqueeze(1) * partials  # [B, n_harmonics, T]

        # Anti-alias
        mask = (f0_bank < self.sr / 2).float()
        c = c * mask

        # Upsample to audio rate
        f0_audio = self.framerate_to_audiorate(f0_bank)                     # [B, n_harmonics, T*hop_length]
        a_audio = self.framerate_to_audiorate(c) * self.attenuate_gain      # [B, n_harmonics, T*hop_length]
        loudness_audio = self.framerate_to_audiorate(a.unsqueeze(1)).squeeze(1)  # [B, T*hop_length]

        # Phase accumulation
        phase = torch.cumsum(f0_audio / self.sr, dim=-1)
        sinusoid_bank = torch.sin(2 * np.pi * phase) * a_audio

        audio = sinusoid_bank.sum(1) * loudness_audio
        return audio
