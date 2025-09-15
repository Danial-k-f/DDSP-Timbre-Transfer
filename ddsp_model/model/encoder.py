import torch
import torchaudio
import torch.nn as nn
from ddsp_model.modules.loudness_extractor import LoudnessExtractor


class Z_Encoder(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        sample_rate=16000,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        z_units=16,
        bidirectional=False,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=20.0,
                f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(
            gru_units * 2 if bidirectional else gru_units, z_units
        )

    def forward(self, audio: torch.Tensor):
        """
        audio: Tensor [B, T]
        returns: Tensor [B, Frames, z_units]
        """
        x = self.mfcc(audio)              # [B, n_mfcc, Frames+1]
        x = x[:, :, :-1]                  # drop last frame for consistency
        x = self.norm(x)
        x = self.permute(x)               # [B, Frames, n_mfcc]
        x, _ = self.gru(x)
        x = self.dense(x)                 # [B, Frames, z_units]
        return x


class Encoder(nn.Module):
    """
    Encoder module with optional Z_Encoder and LoudnessExtractor.

    Input: dict with
        - "audio": Tensor [B, T]
        - "f0": Tensor [B, Frames]

    Output: dict with
        - "loudness": Tensor [B, Frames]
        - "f0": same as input
        - "z": Tensor [B, Frames, z_units] (if use_z=True)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hop_length = int(config.sample_rate * config.frame_resolution)

        # Loudness extractor (ONNX-friendly: returns tuple)
        self.loudness_extractor = LoudnessExtractor(
            sr=config.sample_rate,
            frame_length=self.hop_length,
        )

        # Optional Z encoder
        if config.use_z:
            self.z_encoder = Z_Encoder(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=self.hop_length,
                n_mels=config.n_mels,
                n_mfcc=config.n_mfcc,
                gru_units=config.gru_units,
                z_units=config.z_units,
                bidirectional=config.bidirectional,
            )
        else:
            self.z_encoder = None

    def forward(self, batch: dict):
        """
        batch: dict with "audio" and "f0"
        """
        audio = batch["audio"]

        # Loudness
        loudness = self.loudness_extractor(audio)    # [B, frames]

        batch["loudness"] = loudness

        # Z encoder (optional)
        if self.z_encoder is not None:
            z = self.z_encoder(audio)
            batch["z"] = z

        # Align sequence lengths if hop_length doesn't divide evenly
        target_len = batch["f0"].shape[-1]
        batch["loudness"] = batch["loudness"][:, :target_len]
        if self.z_encoder is not None:
            batch["z"] = batch["z"][:, :target_len]

        return batch
