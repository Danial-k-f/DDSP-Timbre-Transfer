import sys, os
import torch
import torchaudio
import glob
import pandas as pd
import torchyin

# ================================================================
# Configuration
# ================================================================
frame_resolution = 0.005          # seconds per frame (just for time axis)
sample_rate = 44100
confidence_threshold = 0.5

base_data = "data/sax"
splits = ["train", "test"]

# ================================================================
# Function to process one dataset split (train/test/valid)
# ================================================================
def process_split(split):
    input_dir = os.path.join(base_data, split)
    output_dir = os.path.join(input_dir, f"f0_{frame_resolution:.3f}")
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, "*.wav"))
    if len(files) == 0:
        print(f"[WARNING] No wav files found in {input_dir}")
        return

    for wav in files:
        print(f"[INFO] Processing {wav}")
        audio, sr = torchaudio.load(wav)

        if sr != sample_rate:
            raise ValueError(f"[ERROR] {wav} has sample rate {sr}, expected {sample_rate}")

        # Convert to mono
        audio = audio.mean(dim=0)

        # Run YIN pitch tracker (torchyin 0.1.3)
        f0 = torchyin.estimate(audio, sample_rate)   # ONLY two args supported

        # Build confidence mask (dummy: 1 if >0 else 0)
        confidence = torch.ones_like(f0)
        confidence[f0 <= 0.0] = 0.0

        # Build time axis based on frame_resolution
        times = torch.arange(len(f0)) * frame_resolution

        # Apply confidence threshold
        f0[confidence < confidence_threshold] = 0.0

        # Save CSV
        df = pd.DataFrame({
            "time": times.numpy(),
            "frequency": f0.numpy(),
            "confidence": confidence.numpy()
        })
        out_csv = os.path.join(output_dir, os.path.basename(wav).replace(".wav", ".f0.csv"))
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved {out_csv}")

# ================================================================
# Main Execution
# ================================================================
if __name__ == "__main__":
    for split in splits:
        print(f"\n=== Processing split: {split} ===")
        process_split(split)
