import torch
import torch.nn as nn

from ddsp_model.modules.harmonic_oscillator import HarmonicOscillator
from ddsp_model.modules.reverb import TrainableFIRReverb
from ddsp_model.modules.filtered_noise import FilteredNoise
from ddsp_model.model.decoder import Decoder
from ddsp_model.model.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.config = config

        hop_length = int(config.sample_rate * config.frame_resolution)

       # Core synthesis modules
        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate,
            frame_resolution=config.frame_resolution
        )
        self.filtered_noise = FilteredNoise(frame_length=int(config.sample_rate * config.frame_resolution))
        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3)


    # ----------------------------------------------------------
    # ✅ Training-friendly forward (dict input, dict output)
    # ----------------------------------------------------------
    def forward(self, batch, add_reverb=True):
        """
        Input batch dictionary:
            - audio : [B, T] waveform
            - f0    : [B, frames] fundamental frequency
        Returns dictionary with:
            - harmonic, noise, audio_synth, audio_reverb (optional)
            - a (amplitude), c (harmonic distribution), H (filter coeffs)
            - f0, loudness, z (latent)
        """
        enc_out = self.encoder(batch)
        f0 = enc_out["f0"]
        loudness = enc_out["loudness"]
        z = enc_out.get(
            "z",
            torch.zeros(f0.size(0), f0.size(1), self.config.z_units, device=f0.device),
        )

        # Decode latent features
        a, c, H, f0_out = self.decoder(f0, loudness, z)

        # --- Align all time-dependent tensors ---
        min_len = min(f0.size(1), a.size(1), c.size(2), H.size(1))
        f0 = f0[:, :min_len]
        a = a[:, :min_len]
        c = c[:, :, :min_len]
        H = H[:, :min_len]

        # Synthesis
        harmonic = self.harmonic_oscillator(f0, a, c)
        noise = self.filtered_noise(H)
        audio_synth = harmonic + noise[:, : harmonic.size(-1)]

        out = {
            "harmonic": harmonic,
            "noise": noise,
            "audio_synth": audio_synth,
            "a": a,
            "c": c,
            "H": H,
            "f0": f0_out,
            "loudness": loudness,
            "z": z,
        }

        if self.config.use_reverb and add_reverb:
            out["audio_reverb"] = self.reverb(audio_synth)

        return out

    # ----------------------------------------------------------
    # ✅ ONNX-friendly forward (tensor input, tuple output)
    # ----------------------------------------------------------
    def onnx_forward(self, audio: torch.Tensor, f0: torch.Tensor, loudness: torch.Tensor):
        """
        Input:
            audio     : [B, T]
            f0        : [B, frames]
            loudness  : [B, frames]
        Returns:
            harmonic, noise, audio_synth, audio_reverb, a, c, H, f0_out, loudness_out, z
        """
        # make sure f0 and loudness lengths match
        assert f0.size(1) == loudness.size(1), \
            f"Mismatch: f0={f0.size()} vs loudness={loudness.size()}"

        z = torch.zeros(f0.size(0), f0.size(1), self.config.z_units, device=f0.device)

        a, c, H, f0_out = self.decoder(f0, loudness, z)

        # Align with decoder outputs
        min_len = min(f0.size(1), a.size(1), c.size(2), H.size(1))
        f0, loudness = f0[:, :min_len], loudness[:, :min_len]
        a, c, H = a[:, :min_len], c[:, :, :min_len], H[:, :min_len]

        harmonic = self.harmonic_oscillator(f0, a, c)

        # keep noise & reverb minimal for ONNX
        noise = torch.zeros_like(harmonic)
        audio_synth = harmonic
        audio_reverb = harmonic

        return (
            harmonic,
            noise,
            audio_synth,
            audio_reverb,
            a,
            c,
            H,
            f0_out,
            loudness,   # ⚠️ اینجا loudness_out برگشت داده میشه
            z,
        )




    # ----------------------------------------------------------
    # ✅ Simple synthesis test (sanity check)
    # ----------------------------------------------------------
    def test_synthesis(self, f0_start=100, f0_end=1000, duration_sec=2.0):
        """
        Generate a simple f0 sweep using only the HarmonicOscillator.
        Useful for verifying synthesis path independently of training.
        """
        sr = self.config.sample_rate
        frame_length = int(sr * self.config.frame_resolution)

        T = int(duration_sec * sr / frame_length)  # number of frames
        B = 1
        N = self.config.n_harmonics

        # f0 sweep
        f0 = torch.linspace(f0_start, f0_end, T).unsqueeze(0)  # [1,T]
        # only first harmonic active
        c = torch.zeros(B, N, T)
        c[:, 0, :] = 1.0
        # unit amplitude
        a = torch.ones(B, T)

        with torch.no_grad():
            audio = self.harmonic_oscillator(f0, a, c)

        audio = audio.squeeze(0)
        return audio / (audio.abs().max() + 1e-7)
