# DDSP-Timbre-Transfer (48kHz)

This repository contains a **Differentiable Digital Signal Processing (DDSP)** model trained for **timbre transformation** at **48kHz**.  
The model enables users to input an audio signal (e.g., guitar, vocal, synth, etc.) and transform its timbre towards a target sound, while preserving the expressive dynamics and temporal structure of the original performance.

>  General-purpose: although this checkpoint was trained on a specific target dataset, the approach is **not tied to one instrument** (e.g., saxophone). By retraining on your own dataset, you can morph any input source into the desired timbre.

---

##  Features
- **Sample Rate:** 48 kHz (high-quality, studio-ready).  
- **End-to-End Timbre Transfer:** Input audio → Extract features (f0, loudness) → Resynthesized output.  
- **DDSP-based synthesis:** Combines harmonic oscillator with learned parameters, ensuring natural and expressive results.  
- **TorchScript Export:** Model is TorchScript-compiled for portability in Python, C++, and JUCE plugins.  
- **Generalizable Pipeline:** Replace dataset to adapt to other instruments/timbres.  

---

## 📂 Repository Structure
```bash
DDSP-Timbre-Transfer/
│
├── ckpt/ # Pretrained TorchScript/DDSP models
│ └── timbre_48k.ts # TorchScript model (48kHz timbre transfer)
│
├── configs/ # Training & model config files
│ └── timbre_48k.yaml # Example configuration
│
├── data_48k/ # Put your data inside this folder
│ └── sax # Example data
│
├── ddsp_model/                   # Core DDSP implementation
│   ├── model/
│   │   ├── autoencoder.py        # AutoEncoder architecture
│   │   ├── autoencoder_wrapper.py# Wrapper for training & inference
│   │   ├── decoder.py            # Decoder module
│   │   ├── encoder.py            # Encoder module
│   │   └── torchscript_wrapper.py# TorchScript-ready wrapper
│   │
│   ├── modules/
│   │   ├── filtered_noise.py     # Filtered noise generator
│   │   ├── harmonic_oscillator.py# Harmonic oscillator module
│   │   ├── loudness_extractor.py # Loudness feature extractor
│   │   ├── pitch_extractor.py    # Pitch (f0) extractor
│   │   ├── reverb.py             # Reverb module
│   │   └── ptcrepe/   
│   │
│   └── training/                    # Utility functions
│       └── ...
│
├── scripts/                      # Inference and conversion scripts
│   ├── test_torchscript.py       # Test inference with TorchScript model
│   ├── export_torchscript.py     # Export PyTorch -> TorchScript
│   ├── extract_f0_crepe.py
│   ├──	test.py
│   └── train.py                  # training entry point
│
├── README.md # Documentation (this file)
└── requirements.txt # Dependencies
```
---

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/DDSP-Timbre-Transfer.git
cd DDSP-Timbre-Transfer
```
2. Set up environment (Python 3.9+ recommended):
```bash
conda create -n ddsp python=3.9
conda activate ddsp
pip install -r requirements.txt
```
---

## Main dependencies:

- torch >= 2.0

- torchaudio

- librosa

- numpy

- soundfile

---
## 🎧 Usage
1. Test inference with TorchScript model

Run inference on your audio file:
```bash
python scripts/test_torchscript.py \
    --input examples/input.wav \
    --output examples/output.wav \
    --ts_model ckpt/timbre_48k.ts \
    --config configs/timbre_48k.yaml \
    --wave_length 768000
```
---
## 🎹 Training Your Own Model

You can adapt the pipeline for your dataset:

1.Prepare dataset:

-train/ and test/ audio folders

-Mono .wav files at 48kHz

2.Modify config:
```bash
train:

  dataset_path: ./data/your_dataset
  sample_rate: 48000
  batch_size: 16
  ...
```

3.Train:
```bash
python train.py --config configs/your_config.yaml
```

4.Export to TorchScript:
```bash
python scripts/export_torchscript.py --ckpt path/to/checkpoint.pth --output ckpt/your_model.ts
```
---
## 📢 Notes
-The provided model is pretrained for demonstration and timbre transfer at 48kHz.

-Results may vary depending on the source audio. For best quality, retrain with your target instrument or timbre dataset.

-Compatible with Python and C++ (LibTorch / JUCE) environments.

---
## 📜 License

This project is released under the MIT License.
Feel free to use and modify for research or personal projects.

 Developed by Danial Kooshki
📧danial.kooshki@gmail.com
📧 info@danialkooshki.com

🌐 [www.danialkooshki.com](https://www.danialkooshki.com)
