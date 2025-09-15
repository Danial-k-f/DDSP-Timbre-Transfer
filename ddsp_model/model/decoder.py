import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# MLP Block
# ================================================================
class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron block.
    Each layer = Linear → LayerNorm → ReLU
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super().__init__()
        layers = []
        in_dim = n_input
        for i in range(n_layer):
            layers.append(nn.Linear(in_dim, n_units))
            layers.append(nn.LayerNorm(n_units))
            layers.append(relu(inplace=inplace))
            in_dim = n_units
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ================================================================
# Decoder
# ================================================================
class Decoder(nn.Module):
    """
    Decoder network for DDSP.
    Inputs:
        f0: [B, T]  fundamental frequency
        loudness: [B, T]  frame loudness
        z: [B, T, z_units] (optional latent)

    Outputs (tuple):
        a: [B, T] amplitude envelope
        c: [B, n_harmonics, T] harmonic distribution
        H: [B, T, n_freq] filtered noise coefficients
        f0_out: [B, T] aligned fundamental frequency
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------------------------
        # Input pathways (MLPs for f0 and loudness, plus z if enabled)
        # ------------------------------------------------
        self.mlp_f0 = MLPBlock(
            n_input=1,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers
        )
        self.mlp_loudness = MLPBlock(
            n_input=1,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers
        )
        if config.use_z:
            self.mlp_z = MLPBlock(
                n_input=config.z_units,
                n_units=config.mlp_units,
                n_layer=config.mlp_layers
            )
            self.num_mlp = 3
        else:
            self.mlp_z = None
            self.num_mlp = 2

        # ------------------------------------------------
        # Temporal modeling (GRU over time)
        # ------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.num_mlp * config.mlp_units,
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        # ------------------------------------------------
        # Post-GRU MLP
        # ------------------------------------------------
        self.mlp_gru = MLPBlock(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )

        # ------------------------------------------------
        # Output heads
        # ------------------------------------------------
        self.dense_harmonic = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_filter = nn.Linear(config.mlp_units, config.n_freq)

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, f0, loudness, z=None):
        """
        Args:
            f0: [B, T]
            loudness: [B, T]
            z: optional [B, T, z_units]

        Returns:
            a, c, H, f0_out
        """
        # ----------------------------
        # Prepare input
        # ----------------------------
        f0 = f0.unsqueeze(-1)          # [B, T, 1]
        loudness = loudness.unsqueeze(-1)  # [B, T, 1]

        # MLP encodings
        latent_f0 = self.mlp_f0(f0)
        latent_loud = self.mlp_loudness(loudness)

        # ----------------------------
        # Optional Z path
        # ----------------------------
        if self.config.use_z and z is not None:
            # Ensure time dimension alignment
            if z.size(1) != f0.size(1):
                min_len = min(z.size(1), f0.size(1))
                z = z[:, :min_len]
                latent_f0 = latent_f0[:, :min_len]
                latent_loud = latent_loud[:, :min_len]
                f0 = f0[:, :min_len]
            latent_z = self.mlp_z(z)
            latent = torch.cat((latent_f0, latent_z, latent_loud), dim=-1)
        else:
            # Align f0 and loudness if needed
            if latent_loud.size(1) != f0.size(1):
                min_len = min(latent_loud.size(1), f0.size(1))
                latent_f0 = latent_f0[:, :min_len]
                latent_loud = latent_loud[:, :min_len]
                f0 = f0[:, :min_len]
            latent = torch.cat((latent_f0, latent_loud), dim=-1)

        # ----------------------------
        # Temporal modeling
        # ----------------------------
        latent, _ = self.gru(latent)      # [B, T, H]
        latent = self.mlp_gru(latent)     # [B, T, mlp_units]

        # ----------------------------
        # Output heads
        # ----------------------------
        amplitude = self.dense_harmonic(latent)  # [B, T, n_harmonics+1]

        # Amplitude envelope
        a = Decoder.modified_sigmoid(amplitude[..., 0])   # [B, T]

        # Harmonic distribution
        c = torch.softmax(amplitude[..., 1:], dim=-1)     # [B, T, n_harmonics]
        c = c.permute(0, 2, 1)                            # [B, n_harmonics, T]

        # Noise filter
        H = Decoder.modified_sigmoid(self.dense_filter(latent))  # [B, T, n_freq]

        return a, c, H, f0.squeeze(-1)

    # ================================================================
    # Custom sigmoid for amplitude stability
    # ================================================================
    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10 scaling
        a = a.mul(2.0)
        a.add_(1e-7)
        return a
