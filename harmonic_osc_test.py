import torch
import torchaudio
from ddsp_model.modules.harmonic_oscillator import HarmonicOscillator

# ============================
# Config
# ============================
sr = 48000
duration_sec = 2.0
frame_length = 240   # hop size (≈5ms)
device = "cpu"

# ============================
# Init HarmonicOscillator
# ============================
osc = HarmonicOscillator(sr=sr, frame_length=frame_length, device=device)

# ============================
# Build test inputs
# ============================
T = int(duration_sec * sr / frame_length)   # number of frames
B, N = 1, 1  # batch=1, only 1 harmonic

# Fundamental sweep from 100Hz → 1000Hz
f0 = torch.linspace(100, 1000, T).unsqueeze(0)  # [1, T]

# Harmonic distribution: only 1 harmonic active
c = torch.ones(B, N, T)  # [1,1,T]

# Amplitude envelope = 1
a = torch.ones(B, T)     # [1, T]

# ============================
# Run synthesis
# ============================
with torch.no_grad():
    audio = osc(f0, c, a)  # [1, samples]
audio = audio.squeeze(0)   # [samples]

# ============================
# Normalize and save
# ============================
audio = audio / audio.abs().max()
torchaudio.save("harmonic_osc_test.wav", audio.unsqueeze(0), sr)

print("[INFO] Saved test file: harmonic_osc_test.wav")
