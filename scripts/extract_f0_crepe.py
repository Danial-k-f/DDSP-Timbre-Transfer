import sys, os
import torch
import torchaudio
import glob
import pandas as pd

# ================================================================
# Setup Python Path so ddsp_model can be imported
# ================================================================
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from ddsp_model.modules.ptcrepe.ptcrepe import crepe

# ================================================================
# Configuration
# ================================================================
frame_resolution = 0.005          # seconds per frame
step_size = int(frame_resolution * 1000)  # in milliseconds
sample_rate = 48000               # << use 48k since data resampled
model_capacity = "tiny"           # "tiny", "small", "medium", "large", "full"
use_viterbi = True

base_data = "data_48k/sax"        # point to your 48k dataset folder
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

    # Initialize CREPE model once per split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = crepe.CREPE(model_capacity).to(device)

    for wav in files:
        print(f"[INFO] Processing {wav}")

        # Load and resample audio if needed
        audio, sr = torchaudio.load(wav)
        if sr != sample_rate:
            print(f"[WARNING] Resampling {wav} from {sr} Hz to {sample_rate} Hz")
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            audio = transform(audio)

        # Convert to mono
        audio = audio.mean(dim=0).unsqueeze(0)

        # Run CREPE prediction
        with torch.no_grad():
            time, frequency, confidence, _ = model.predict(
                audio,
                sample_rate,
                step_size=step_size,
                viterbi=use_viterbi,
                batch_size=8
            )

        # Save output to CSV
        df = pd.DataFrame({
            "time": time.cpu().numpy(),
            "frequency": frequency.cpu().numpy(),
            "confidence": confidence.cpu().numpy()
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
