import torch
import torchaudio
from ddsp_model.model.autoencoder import AutoEncoder

# Dummy config (align with training config)
class Config:
    sample_rate = 48000
    frame_resolution = 0.005
    n_harmonics = 101
    n_freq = 65
    gru_units = 512
    mlp_units = 512
    mlp_layers = 3
    use_z = False
    z_units = 16
    bidirectional = False
    n_fft = 2048
    n_mels = 128
    n_mfcc = 30
    use_reverb = True

config = Config()

# Init model
model = AutoEncoder(config)

# Run test synthesis
audio = model.test_synthesis(f0_start=100, f0_end=1000, duration_sec=3.0)

# Save
torchaudio.save("quick_test.wav", audio.unsqueeze(0), config.sample_rate)
print("[INFO] Saved quick_test.wav")
