import torch
import os, sys
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from ddsp_model.model.autoencoder import AutoEncoder
from ddsp_model.model.torchscript_wrapper import TorchScriptWrapper

# ------------------------------
# مسیرها
# ------------------------------
config_path = "E:/ddsp-pytorch-master/DDSP/configs/sax_48k.yaml"
ckpt_path   = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_final.pth"
out_path    = "E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k.ts"

config = OmegaConf.load(config_path)
device = "cpu"

# ------------------------------
# مدل اصلی
# ------------------------------
model = AutoEncoder(config).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("[INFO] Model loaded")

# ------------------------------
# Wrapper برای خروجی tuple
# ------------------------------
script_model = TorchScriptWrapper(model).to(device)

# ------------------------------
# ورودی ساختگی
# ------------------------------
B = 1
T = int(config.sample_rate * config.waveform_sec)
frames = int(T / int(config.sample_rate * config.frame_resolution))

dummy_audio = torch.randn(B, T, device=device)
dummy_f0 = torch.rand(B, frames, device=device) * 440
dummy_batch = {"audio": dummy_audio, "f0": dummy_f0}

# ------------------------------
# Export
# ------------------------------
scripted = torch.jit.trace(script_model, (dummy_batch,))
scripted.save(out_path)
print(f"[INFO] ✅ TorchScript model saved: {out_path}")
