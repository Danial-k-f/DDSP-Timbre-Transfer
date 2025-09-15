import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# ================================================================
# Configuration
# ================================================================
# Path to one of the extracted CSV files (change this to your own)
csv_path = "../data_48k/sax/train/f0_0.005/sax2.f0.csv"

# ================================================================
# Load CSV
# ================================================================
if not os.path.exists(csv_path):
    print(f"[ERROR] File not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

# ================================================================
# Plot F0 over time
# ================================================================
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["frequency"], label="F0 (Hz)", color="b")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Fundamental Frequency (F0) from {os.path.basename(csv_path)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
