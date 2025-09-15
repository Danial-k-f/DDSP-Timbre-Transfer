import torch
import torchaudio
import onnxruntime as ort
import soundfile as sf
import numpy as np
from ddsp_model.modules.ptcrepe.ptcrepe.crepe import CREPE
from ddsp_model.modules.loudness_extractor import LoudnessExtractor

# --------------------------------------------------------
# Load WAV
# --------------------------------------------------------
wav_path = "C:/Users/dania/OneDrive/Desktop/Data/gtr.wav"
y, sr = torchaudio.load(wav_path)   # [C, T]
y = y.mean(dim=0, keepdim=True)     # mono [1, T]
print(f"[INFO] Loaded WAV: {wav_path}, sr={sr}, length={y.shape[1]}")

# --------------------------------------------------------
# Make two versions: 16kHz (CREPE) and 48kHz (Loudness + Model)
# --------------------------------------------------------
y_crepe = torchaudio.functional.resample(y, orig_freq=sr, new_freq=16000)
y_model = torchaudio.functional.resample(y, orig_freq=sr, new_freq=48000)

print(f"[INFO] Resampled for CREPE: {y_crepe.shape[1]} samples @16k")
print(f"[INFO] Resampled for Model: {y_model.shape[1]} samples @48k")

# --------------------------------------------------------
# Device setup
# --------------------------------------------------------
device = "cpu"
print(f"[INFO] Using device for CREPE: {device}")

# --------------------------------------------------------
# F0 extraction with CREPE
# --------------------------------------------------------
f0_extractor = CREPE("tiny").to(device)

hop = int(sr * 0.005)  # 5 ms hop

pad_len = hop - (y_model.shape[1] % hop)
if pad_len != hop:
    y_model = torch.nn.functional.pad(y_model, (0, pad_len))

with torch.no_grad():
    time, f0, confidence, _ = f0_extractor.predict(
        y_model.to(device), sr=sr, viterbi=True, step_size=5, batch_size=8
    )

f0 = f0.unsqueeze(0).float().cpu()
confidence = torch.tensor(confidence).unsqueeze(0)
f0[confidence < 0.5] = 0.0
print(f"[INFO] Extracted f0 shape {f0.shape}")

# --------------------------
# Loudness extraction (also 5ms)
# --------------------------
hop_model = int(sr * 0.005)   # 5ms hop @48k
pad_len = hop_model - (y_model.shape[1] % hop_model)
if pad_len != hop_model:
    y_model = torch.nn.functional.pad(y_model, (0, pad_len))

loud_extractor = LoudnessExtractor(sr=sr, frame_length=hop_model).to(device)
with torch.no_grad():
    loudness = loud_extractor(y_model.to(device)).cpu()
print(f"[INFO] Extracted loudness shape {loudness.shape}")
# --------------------------------------------------------
# Load ONNX model
# --------------------------------------------------------
onnx_path = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_last.onnx"
print("[INFO] Loading ONNX model...")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print("[INFO] Model loaded.")

# --------------------------------------------------------
# Align and Run Inference
# --------------------------------------------------------
min_len = min(f0.shape[1], loudness.shape[1])
f0 = f0[:, :min_len]
loudness = loudness[:, :min_len]

f0_np = f0.numpy().astype("float32")
loudness_np = loudness.numpy().astype("float32")

inputs = {
    "f0": f0_np,
    "loudness": loudness_np
}

print("[INFO] Running inference...")
outputs = session.run(None, inputs)
harmonic, noise, audio_synth, audio_reverb = outputs[:4]

print("[INFO] Inference finished.")
print(f" - harmonic: {harmonic.shape}")
print(f" - audio_reverb: {audio_reverb.shape}")

# --------------------------------------------------------
# Save audio (48kHz to match model)
# --------------------------------------------------------
out_audio = audio_reverb.squeeze()
out_audio = out_audio / (np.abs(out_audio).max() + 1e-8)
out_path = "C:/Users/dania/OneDrive/Desktop/Data/gtr_saxx_48.wav"
sf.write(out_path, out_audio, 48000)
print(f"[INFO] ✅ Saved output audio to {out_path}")
