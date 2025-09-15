import torch
from omegaconf import OmegaConf
from ddsp_model.model.autoencoder import AutoEncoder

# ------------------------------
# ONNX Wrapper
# ------------------------------
class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, f0, loudness):
        return self.model.onnx_forward(audio, f0, loudness)


# ------------------------------
# Config & Checkpoint paths
# ------------------------------
config_path = "E:/ddsp-pytorch-master/DDSP/configs/sax_48k.yaml"
ckpt_path = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_last.pth"
onnx_path = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_last.onnx"

config = OmegaConf.load(config_path)
device = "cpu"   # ONNX export must be on CPU

# ------------------------------
# Load model
# ------------------------------
model = AutoEncoder(config).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

onnx_model = ONNXWrapper(model)
print("[INFO] Model loaded for ONNX export")

# ------------------------------
# Dummy inputs for tracing
# ------------------------------
B = 1
T = int(config.sample_rate * config.waveform_sec)  # 1 sec by default
frames = int(T / int(config.sample_rate * config.frame_resolution))

dummy_audio = torch.randn(B, T, device=device)
dummy_f0 = torch.rand(B, frames, device=device) * 440  # fake f0 [Hz]
dummy_loudness = torch.rand(1, frames, device=device)


# ------------------------------
# Export
# ------------------------------
torch.onnx.export(
    onnx_model,
    (dummy_audio, dummy_f0, dummy_loudness),
    onnx_path,
    input_names=["audio", "f0", "loudness"],
    output_names=["harmonic", "noise", "audio_synth", "audio_reverb",
                  "a", "c", "H", "f0_out", "loudness_out", "z"],
    dynamic_axes={
        "audio": {1: "audio_len"},
        "f0": {1: "frame_len"},
        "loudness": {1: "frame_len"},
        "harmonic": {1: "audio_len"},
        "noise": {1: "audio_len"},
        "audio_synth": {1: "audio_len"},
        "audio_reverb": {1: "audio_len"},
    },
    opset_version=17,
    verbose=True
    
)
print(f"[INFO] ✅ ONNX model exported successfully: {onnx_path}")

