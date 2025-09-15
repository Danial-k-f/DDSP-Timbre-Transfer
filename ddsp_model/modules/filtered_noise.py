import torch
import torch.nn as nn
import torch.fft


class FilteredNoise(nn.Module):
    def __init__(self, frame_length=64, attenuate_gain=1e-2):
        super().__init__()
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: [B, T, F]  frequency magnitude response
        Returns:
            out: [B, T * frame_length]  time-domain filtered noise
        """
        B, T, F_bins = H.shape
        device = H.device

        # (1) Treat H as the real part, imag = 0
        H_real = H.reshape(B * T, F_bins)
        H_imag = torch.zeros_like(H_real)

        # (2) iFFT of the spectrum (manual real+imag multiplication is safe for ONNX)
        #     Here we use irfft directly with real only (since imag=0)
        h_time = torch.fft.irfft(H_real, n=F_bins * 2 - 1)   # [B*T, L]

        # Hann window to smooth
        win = torch.hann_window(h_time.shape[-1], device=device)
        h_time = h_time * win

        # (3) FFT of filter (real only)
        L = h_time.shape[-1] + self.frame_length - 1
        H_f = torch.fft.rfft(h_time, n=L)                    # [B*T, L//2+1]

        # (4) White noise
        noise = torch.rand(B * T, self.frame_length, device=device) * 2 - 1
        noise_f = torch.fft.rfft(noise, n=L)                 # [B*T, L//2+1]

        # (5) Multiply in frequency domain (manually safe)
        # ONNX can export .real and .imag separately
        Hr, Hi = H_f.real, H_f.imag
        Nr, Ni = noise_f.real, noise_f.imag

        R = Nr * Hr - Ni * Hi
        I = Nr * Hi + Ni * Hr

        # (6) Back to time domain
        filtered = torch.fft.irfft(torch.complex(R, I), n=L) # [B*T, L]

        # (7) Reshape → overlap add
        filtered = filtered.view(B, T, -1) * self.attenuate_gain
        out = filtered.reshape(B, -1)[:, : T * self.frame_length]

        return out
