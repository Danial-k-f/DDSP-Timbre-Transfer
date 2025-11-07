from cog import BasePredictor, Input, Path
import os
import torch
import torchaudio
from omegaconf import OmegaConf

# -----------------------------
# Model paths
# -----------------------------
CKPT_PATH   = "exported_model/sax_48k_last.pth"
CONFIG_PATH = "configs/sax_48k.yaml"


class Predictor(BasePredictor):
    def setup(self):
        """Load DDSP model and feature extractors once"""
        from ddsp_model.model.autoencoder_wrapper import AutoEncoderWrapper
        from ddsp_model.modules.loudness_extractor import LoudnessExtractor

        # Load config
        self.config = OmegaConf.load(CONFIG_PATH)
        self.target_sr = int(self.config.sample_rate)

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Load model
        self.wrapper = AutoEncoderWrapper(self.config, device=self.device)
        state = torch.load(CKPT_PATH, map_location=self.device)

        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        self.wrapper.model.load_state_dict(state)
        self.wrapper.model.eval()

        # Load loudness extractor
        self.loud_extractor = LoudnessExtractor(self.target_sr).to(self.device)
        print("[INFO] Model and loudness extractor loaded successfully.")

    def predict(
        self,
        audio: Path = Input(description="Input WAV file"),
        add_reverb: bool = Input(description="Apply model reverb", default=True),
        wave_length: int = Input(description="Set 0 for full audio or number of frames", default=0),
    ) -> Path:
        """Run DDSP inference on input audio"""

        # --- Fix wave_length input type (prevents NoneType comparison bug) ---
        if wave_length in [None, "None", "", "0"]:
            wave_length = 0
        else:
            try:
                wave_length = int(float(wave_length))
            except Exception:
                wave_length = 0

        print(f"[INFO] Loading audio (wave_length={wave_length})")

        # --- Load input audio ---
        y, sr = torchaudio.load(str(audio),
                                num_frames=None if wave_length == 0 else wave_length)

        # Convert stereo → mono
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            y = torchaudio.functional.resample(y, sr, self.target_sr)
            sr = self.target_sr

        y = y.to(self.device)

        # --- Feature extraction ---
        print("[INFO] Extracting f0 and loudness...")
        f0 = self.wrapper.get_f0(y, sample_rate=sr)

        with torch.no_grad():
            loudness = self.loud_extractor(y)

        # --- Reconstruction ---
        print("[INFO] Running model inference...")
        with torch.no_grad():
            outputs = self.wrapper.reconstruction(
                y,
                f0=f0,
                loudness=loudness,
                add_reverb=add_reverb if hasattr(self.config, "use_reverb") else add_reverb
            )

        # Choose output (with or without reverb)
        if getattr(self.config, "use_reverb", False) and "audio_reverb" in outputs:
            out = outputs["audio_reverb"]
        else:
            out = outputs["audio_synth"]

        # [B,1,T] → [B,T]
        if out.dim() == 3 and out.size(1) == 1:
            out = out.squeeze(1)

        # --- Save output ---
        out_path = "sax_output.wav"
        torchaudio.save(out_path, out.cpu(), sample_rate=self.target_sr)
        print(f"[INFO] Output saved to {out_path}")

        return Path(out_path)
