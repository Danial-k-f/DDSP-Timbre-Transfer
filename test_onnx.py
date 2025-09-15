import onnxruntime as ort
import numpy as np

# -------------------------------------------------
# Paths
# -------------------------------------------------
onnx_path = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_last.onnx"

# -------------------------------------------------
# Load ONNX model
# -------------------------------------------------
print("[INFO] Loading ONNX model...")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print("[INFO] Model loaded.")

# -------------------------------------------------
# Inspect inputs/outputs
# -------------------------------------------------
print("\nInputs:")
for inp in session.get_inputs():
    print(f" - {inp.name}: {inp.shape}, {inp.type}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f" - {out.name}: {out.shape}, {out.type}")

# -------------------------------------------------
# Dummy input data (1 sec audio at 48k, f0 frames)
# -------------------------------------------------
sr = 48000
T = sr  # 1 second
frames = int(T / (sr * 0.005))  # assuming frame_resolution=0.005

dummy_audio = np.random.randn(1, T).astype(np.float32)
dummy_f0 = np.linspace(100, 300, frames, dtype=np.float32).reshape(1, -1)
dummy_loudness = np.random.rand(1, frames).astype(np.float32)

# -------------------------------------------------
# Run inference
# -------------------------------------------------
inputs = {
    "f0": dummy_f0,
    "loudness.1": dummy_loudness,
}

print("\n[INFO] Running inference...")
outputs = session.run(None, inputs)


# -------------------------------------------------
# Print shapes
# -------------------------------------------------
print("\nOutput shapes:")
for name, out in zip([o.name for o in session.get_outputs()], outputs):
    print(f" - {name}: {np.array(out).shape}")

print("\n[INFO] ✅ Test finished successfully.")
