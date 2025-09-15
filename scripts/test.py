import torch
import torchaudio
import os, sys
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from omegaconf import OmegaConf
from ddsp_model.model.autoencoder_wrapper import AutoEncoderWrapper
from ddsp_model.modules.loudness_extractor import LoudnessExtractor

# ------------------------------
# Argparse
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--ckpt", default=".pth")
parser.add_argument("--config", default=".yaml")
parser.add_argument("--wave_length", default=0, type=int,
                    help="0 means load full wav, may cause GPU OOM")
args = parser.parse_args()

# ------------------------------
# Load input wav
# ------------------------------
y, sr = torchaudio.load(
    args.input,
    num_frames=None if args.wave_length == 0 else args.wave_length
)
print(f"[INFO] File loaded: {args.input} | sr={sr}, length={y.shape[-1]}")

# ------------------------------
# Load config
# ------------------------------
config = OmegaConf.load(args.config)
target_sr = int(config.sample_rate)

# Resample if needed
if sr != target_sr:
    print(f"[INFO] Resampling {sr} -> {target_sr}")
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    y = resampler(y)
    sr = target_sr

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load model + ckpt
# ------------------------------
wrapper = AutoEncoderWrapper(config, device=device)
wrapper.model.load_state_dict(torch.load(args.ckpt, map_location=device))
wrapper.model.eval()
print("[INFO] Network Loaded")

# ------------------------------
# Feature extraction
# ------------------------------
# f0
f0 = wrapper.get_f0(y.to(device), sample_rate=sr)

# loudness
loud_extractor = LoudnessExtractor(int(config.sample_rate)).to(device)
with torch.no_grad():
    loudness = loud_extractor(y.to(device))

# ------------------------------
# Reconstruction
# ------------------------------
with torch.no_grad():
    outputs = wrapper.reconstruction(y.to(device), f0=f0, add_reverb=config.use_reverb)

# ------------------------------
# Save synth
# ------------------------------
synth = outputs["audio_synth"]
if synth.dim() == 3:   # [B, 1, T]
    synth = synth.squeeze(1)  # -> [B, T]
torchaudio.save(
    os.path.splitext(args.output)[0] + "_synth.wav",
    synth.cpu(),
    sample_rate=target_sr
)

# ------------------------------
# Save reverb if available
# ------------------------------
if config.use_reverb and "audio_reverb" in outputs:
    reverb = outputs["audio_reverb"]
    if reverb.dim() == 3:
        reverb = reverb.squeeze(1)
    torchaudio.save(
        os.path.splitext(args.output)[0] + "_reverb.wav",
        reverb.cpu(),
        sample_rate=target_sr
    )

print("[INFO] Test completed. Files saved.")
