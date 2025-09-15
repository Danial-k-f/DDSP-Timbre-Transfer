import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableFIRReverb(nn.Module):
    def __init__(self, reverb_length=48000, device="cpu"):
        """
        Simple trainable FIR reverb model (ONNX-friendly).
        
        Args:
            reverb_length: Length of the reverb impulse response
            device: "cpu" or "cuda"
        """
        super().__init__()
        self.reverb_length = reverb_length
        self.device = device

        # Trainable FIR (initialized in [-1, 1])
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32) * 2 - 1,
            requires_grad=True,
        )

        # Trainable dry/wet mix parameter
        self.drywet = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)

        # Trainable decay parameter
        self.decay = nn.Parameter(torch.tensor([3.0], dtype=torch.float32), requires_grad=True)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] dry audio signal
        Returns:
            output: [B, T + reverb_length - 1] wet audio signal
        """
        B, T = audio.shape

        # Build decay envelope
        decay_env = torch.exp(
            -(torch.exp(self.decay) + 2.0)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32, device=audio.device)
        )

        # Apply decay to FIR
        decay_fir = self.fir * decay_env.unsqueeze(0)

        # Identity impulse (for dry mix)
        ir_identity = torch.zeros_like(decay_fir)
        ir_identity[:, 0] = 1.0

        # Mix dry and wet impulse responses
        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir
            + (1.0 - torch.sigmoid(self.drywet)) * ir_identity
        )  # [1, L]

        # Convolution via FFT (ONNX-friendly)
        n_fft = T + self.reverb_length - 1
        audio_f = torch.fft.rfft(audio, n=n_fft)          # [B, F]
        fir_f = torch.fft.rfft(final_fir, n=n_fft)        # [1, F]

        output_f = audio_f * fir_f                        # broadcasting over batch
        output = torch.fft.irfft(output_f, n=n_fft)       # [B, T + L - 1]

        return output
