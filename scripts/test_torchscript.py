import torch
import torchaudio
import os, sys
import argparse
import soundfile as sf

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from omegaconf import OmegaConf
from ddsp_model.modules.loudness_extractor import LoudnessExtractor
from ddsp_model.modules.ptcrepe.ptcrepe.crepe import CREPE

# ------------------------------
# Argparse
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav")
parser.add_argument("--output", default="output.wav")
parser.add_argument("--ts_model", default=".ts")
parser.add_argument("--config", default=".yaml")
parser.add_argument("--wave_length", default=0, type=int)
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

device = "cpu"   # ⚠️ همه چیز روی CPU

# ------------------------------
# Load TorchScript model
# ------------------------------
model = torch.jit.load(args.ts_model, map_location=device)
model.eval()
print("[INFO] TorchScript model loaded")

# ------------------------------
# Feature extraction
# ------------------------------
crepe = CREPE("tiny").to(device)
with torch.no_grad():
    time, f0, confidence, _ = crepe.predict(
        y.to("cpu"),   # ⚠️ همیشه روی CPU
        sr=sr,
        viterbi=True,
        step_size=int(config.frame_resolution * 1000),
        batch_size=8
    )

# f0 → [1, T]
f0 = f0.float().unsqueeze(0).to(device)
confidence = torch.tensor(confidence, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
f0[confidence < 0.5] = 0.0
print(f"[INFO] Extracted f0 shape {f0.shape}")

# loudness
hop_length = int(sr * config.frame_resolution)
loud_extractor = LoudnessExtractor(sr=sr, frame_length=hop_length).to(device)
with torch.no_grad():
    loudness = loud_extractor(y.to(device)).cpu()
loudness = loudness.to(device)
print(f"[INFO] Extracted loudness shape {loudness.shape}")

# ------------------------------
# Align and Run Inference
# ------------------------------
min_len = min(f0.shape[1], loudness.shape[1])
f0 = f0[:, :min_len]
loudness = loudness[:, :min_len]

with torch.no_grad():
    batch = {
        "audio": y.to(device),
        "f0": f0,
        "loudness": loudness
    }
    outputs = model(batch)

# ------------------------------
# Save audio
# ------------------------------
harmonic, noise, audio_synth, audio_reverb, *_ = outputs
out_audio = audio_reverb.squeeze().cpu().numpy()
out_audio = out_audio / (abs(out_audio).max() + 1e-8)

sf.write(args.output, out_audio, target_sr)
print(f"[INFO] ✅ Saved output audio to {args.output}")
