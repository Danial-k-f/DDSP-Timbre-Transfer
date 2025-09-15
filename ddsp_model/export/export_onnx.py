import torch
from ddsp.model.autoencoder import AutoEncoder
import onnx

def export_onnx(config, ckpt_path, out_path="sax_44k.onnx"):
    model = AutoEncoder(config).eval()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'])

    dummy_audio = torch.randn(1, 44100)    # 1 sec
    dummy_f0 = torch.randn(1, 200)
    dummy_loud = torch.randn(1, 200)

    torch.onnx.export(
        model,
        (dummy_audio, dummy_f0, dummy_loud),
        out_path,
        input_names=["audio", "f0", "loudness"],
        output_names=["audio_synth", "harmonic", "noise", "a", "c", "H", "audio_reverb"],
        opset_version=17,
    )
    print(f"ONNX model saved at {out_path}")
