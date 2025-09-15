import torch
from ddsp_model.model.autoencoder import AutoEncoder


class AutoEncoderWrapper:
    def __init__(self, config, device="cpu"):
        self.device = device
        self.model = AutoEncoder(config).to(device)
        self.config = config
        self.crepe = None

    def get_f0(self, audio, sample_rate=None, f0_threshold=0.5):
        """
        Estimate F0 using CREPE. Always returns shape [1, T].
        """
        if self.crepe is None:
            from ddsp_model.modules.ptcrepe.ptcrepe.crepe import CREPE
            self.crepe = CREPE(self.config.crepe).to(self.device)

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # CREPE همیشه روی CPU کار می‌کنه → بعد برمی‌گردونیم به device
        with torch.no_grad():
            time, f0, confidence, _ = self.crepe.predict(
                audio.cpu(),
                sr=sample_rate,
                viterbi=True,
                step_size=int(self.config.frame_resolution * 1000),
                batch_size=32
            )
            f0 = f0.float()
            f0[confidence < f0_threshold] = 0.0
            f0 = f0.unsqueeze(0)  # [1, T]
            return f0.to(self.device)

    def reconstruction(self, audio, f0=None, add_reverb=True):
        self.model.eval()
        with torch.no_grad():
            if f0 is None:
                f0 = self.get_f0(audio)
    
            # Ensure audio shape is [1, T]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 2 and audio.size(0) > 1:
                audio = audio[:1, :]  # take first channel
    
            # ---- ✅ از forward اصلی استفاده کن ----
            batch = {"audio": audio.to(self.device), "f0": f0}
            outputs = self.model(batch, add_reverb=add_reverb)
    
            return outputs

