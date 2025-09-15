import torch
import torch.nn as nn


class LoudnessExtractor(nn.Module):
    def __init__(self, sr=48000, frame_length=240, attenuate_gain=2.0, device="cpu"):
        super().__init__()
        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = frame_length * 5
        self.device = device
        self.attenuate_gain = attenuate_gain

        # Hann window
        self.register_buffer(
            "smoothing_window", torch.hann_window(self.n_fft, dtype=torch.float32)
        )

        # Frequency bins
        freqs = torch.linspace(0, sr // 2, self.n_fft // 2 + 1)
        self.register_buffer("freqs", freqs)

        # A-weighting
        self.register_buffer("a_weights", self.torch_A_weighting(freqs))

    def torch_A_weighting(self, freqs: torch.Tensor, min_db: float = -45.0) -> torch.Tensor:
        f_sq = freqs**2
        const = torch.tensor([12200, 20.6, 107.7, 737.9], dtype=torch.float32) ** 2

        weights_db = (
            2.0
            + 20.0
            * (
                torch.log10(const[0])
                + 4 * torch.log10(freqs + 1e-6)
                - torch.log10(f_sq + const[0])
                - torch.log10(f_sq + const[1])
                - 0.5 * torch.log10(f_sq + const[2])
                - 0.5 * torch.log10(f_sq + const[3])
            )
        )
        weights_db = torch.clamp(weights_db, min=min_db)
        return 10.0 ** (weights_db / 20.0)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T]
        Returns:
            loudness: [B, frames]
        """
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.frame_length,
            window=self.smoothing_window.to(audio.device),
            return_complex=False,  # ✅ keep real+imag split
        )  # [B, F, frames, 2]

        real = stft[..., 0]
        imag = stft[..., 1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)  # ✅ prevent NaN

        power = mag**2

        # Apply A-weighting
        weights = self.a_weights[: power.size(1)].to(audio.device).unsqueeze(0).unsqueeze(-1)
        weighted = power * weights

        # Loudness
        loudness = torch.sqrt(weighted.sum(dim=1)) / self.n_fft * self.attenuate_gain
        if loudness.dim() == 1:
            loudness = loudness.unsqueeze(0)

        return loudness

