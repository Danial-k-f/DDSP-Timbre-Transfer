import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import onnxruntime as ort
from ddsp_model.modules.loudness_extractor import LoudnessExtractor

# --------------------------------------------------------
# Load WAV
# --------------------------------------------------------
wav_path = "C:/Users/dania/OneDrive/Desktop/Data/Ex79.wav"
y, sr = torchaudio.load(wav_path)   # [C, T]
y = y.mean(dim=0).numpy()           # mono [T]
print(f"[INFO] Loaded WAV: {wav_path}, sr={sr}, length={len(y)}")

# --------------------------------------------------------
# Resample to model rate (48kHz)
# --------------------------------------------------------
target_sr = 48000
if sr != target_sr:
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sr = target_sr
print(f"[INFO] Resampled audio to {sr} Hz, new length={len(y)}")

# --------------------------------------------------------
# F0 extraction with librosa.pyin (lightweight)
# --------------------------------------------------------
hop_length = int(sr * 0.005)  # 5ms hop
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=50, fmax=2000,
    sr=sr,
    frame_length=2048,
    hop_length=hop_length
)

# Replace NaN with 0
f0 = np.nan_to_num(f0, nan=0.0)
f0 = torch.tensor(f0, dtype=torch.float32).unsqueeze(0)  # [1, frames]
print(f"[INFO] Extracted f0 shape {f0.shape}")

# --------------------------------------------------------
# Loudness extraction (same hop)
# --------------------------------------------------------
y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
pad_len = hop_length - (y_torch.shape[1] % hop_length)
if pad_len != hop_length:
    y_torch = torch.nn.functional.pad(y_torch, (0, pad_len))

loud_extractor = LoudnessExtractor(sr=sr, frame_length=hop_length)
with torch.no_grad():
    loudness = loud_extractor(y_torch).cpu()
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

inputs = {
    "f0": f0.numpy().astype("float32"),
    "loudness": loudness.numpy().astype("float32"),
}

print("[INFO] Running inference...")
outputs = session.run(None, inputs)
harmonic, noise, audio_synth, audio_reverb = outputs[:4]

print("[INFO] Inference finished.")
print(f" - harmonic: {harmonic.shape}")
print(f" - audio_reverb: {audio_reverb.shape}")

# --------------------------------------------------------
# Save audio
# --------------------------------------------------------
out_audio = audio_reverb.squeeze()
out_audio = out_audio / (np.abs(out_audio).max() + 1e-8)
out_path = "C:/Users/dania/OneDrive/Desktop/Data/Ex79_pyinf0.wav"
sf.write(out_path, out_audio, sr)
print(f"[INFO] ✅ Saved output audio to {out_path}")
