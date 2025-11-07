
from cog import BasePredictor, Input, Path
import os
import torch
import torchaudio
from omegaconf import OmegaConf


CKPT_PATH   = "exported_model/sax_48k_last.pth" 
CONFIG_PATH = "configs/sax_48k.yaml"               

class Predictor(BasePredictor):
    def setup(self):
   
        from ddsp_model.model.autoencoder_wrapper import AutoEncoderWrapper
        from ddsp_model.modules.loudness_extractor import LoudnessExtractor

        self.config = OmegaConf.load(CONFIG_PATH)
        self.target_sr = int(self.config.sample_rate)

        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.wrapper = AutoEncoderWrapper(self.config, device=device)
        state = torch.load(CKPT_PATH, map_location=device)
        
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.wrapper.model.load_state_dict(state)
        self.wrapper.model.eval()

        
        self.loud_extractor = LoudnessExtractor(self.target_sr).to(device)
        print("[INFO] Model & extractor loaded.")

    def predict(
        self,
        audio: Path = Input(description="Input WAV file"),
        add_reverb: bool = Input(description="Apply model reverb", default=True),
        wave_length: int = Input(description="0=full; truncate frames for testing", default=0),
    ) -> Path:
        
        y, sr = torchaudio.load(str(audio), num_frames=None if wave_length == 0 else wave_length)
       
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        
        if sr != self.target_sr:
            y = torchaudio.functional.resample(y, sr, self.target_sr)
            sr = self.target_sr

        y = y.to(self.device)

        
        # f0
        f0 = self.wrapper.get_f0(y, sample_rate=sr)

        # loudness 
        with torch.no_grad():
            loudness = self.loud_extractor(y) 

        # --- Reconstruction ---
        with torch.no_grad():
            outputs = self.wrapper.reconstruction(
                y, f0=f0, add_reverb=add_reverb if hasattr(self.config, "use_reverb") else add_reverb
            )

       
        if (getattr(self.config, "use_reverb", False) and "audio_reverb" in outputs):
            out = outputs["audio_reverb"]
        else:
            out = outputs["audio_synth"]

        # [B,1,T] -> [B,T]
        if out.dim() == 3 and out.size(1) == 1:
            out = out.squeeze(1)

        # SAVE
        out_path = "sax_output.wav"
        torchaudio.save(out_path, out.cpu(), sample_rate=self.target_sr)
        return Path(out_path)
