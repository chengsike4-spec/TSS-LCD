import torch.nn as nn



# ================== Future Autoencoder (保持原始结构) ==================
# class FutureAutoencoder(nn.Module):
#     def __init__(self, seq_len, D, latent_dim):
#         super().__init__()
#         self.seq_len, self.D = seq_len, D
#         self.input_dim = seq_len * D
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(self.input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, latent_dim)
#         )
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, self.input_dim)
#         )
#
#     def forward(self, x):
#         B = x.size(0)
#         flat = x.view(B, -1)
#         z = self.encoder(flat)
#         recon_flat = self.decoder(z)
#         recon = recon_flat.view(B, self.seq_len, self.D)
#         return recon, z

# class FutureAutoencoder(nn.Module):
#     def __init__(self, seq_len, D, latent_dim):
#         super().__init__()
#         self.seq_len, self.D = seq_len, D
#         self.input_dim = seq_len * D
#
#         self.encoder = nn.Sequential(
#         nn.LayerNorm([seq_len, D]),
#         nn.Conv1d(D, 128, kernel_size=3, padding=1), # shape: (B, 128, T)
#         nn.ReLU(),
#         nn.Conv1d(128, 64, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Flatten(), # shape: (B, T*64)
#         nn.Linear(seq_len * 64, latent_dim)
#         )
#
#         self.decoder = nn.Sequential(
#         nn.Linear(latent_dim, seq_len * 64),
#         nn.Unflatten(1, (64, seq_len)),
#         nn.ConvTranspose1d(64, 128, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.ConvTranspose1d(128, D, kernel_size=3, padding=1),
#         nn.Tanh()
#         )
#
#     def forward(self, x):
#         # x: (B, T, D) → transpose to (B, D, T) for conv1d
#         x = x.transpose(1, 2)
#         z = self.encoder(x)
#         recon = self.decoder(z) # (B, D, T)
#         recon = recon.transpose(1, 2)
#         return recon, z

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual

class FutureAutoencoder(nn.Module):
    def __init__(self, seq_len, D, latent_dim, hidden_dim=1024, num_blocks=3):
        super().__init__()
        self.seq_len, self.D = seq_len, D
        self.input_dim = seq_len * D

        # 编码器
        self.encoder_input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU()
        )
        self.encoder_blocks = nn.Sequential(
            *[MLPBlock(hidden_dim, hidden_dim * 2) for _ in range(num_blocks)]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder_input = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU()
        )
        self.decoder_blocks = nn.Sequential(
            *[MLPBlock(hidden_dim, hidden_dim * 2) for _ in range(num_blocks)]
        )
        self.to_output = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        x = self.encoder_input(flat)
        x = self.encoder_blocks(x)
        z = self.to_latent(x)

        y = self.decoder_input(z)
        y = self.decoder_blocks(y)
        recon_flat = self.to_output(y)
        recon = recon_flat.view(B, self.seq_len, self.D)
        return recon, z

# class FutureAutoencoder(nn.Module):
#     def __init__(self, seq_len, D, latent_dim, L=3, F=250, hidden_dim=32, num_blocks=1):
#         super().__init__()
#         self.seq_len = seq_len
#         self.D = D
#         self.latent_dim = latent_dim
#         self.L = L
#         self.F = F
#         assert L * F == D, f"L×F 必须等于 D（got {L}×{F} = {L*F} ≠ {D})"
#
#         # 编码器
#         encoder_layers = []
#         in_channels = 1
#         for _ in range(num_blocks):
#             encoder_layers += [
#                 nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm3d(hidden_dim),
#                 nn.ReLU()
#             ]
#             in_channels = hidden_dim
#         self.encoder3d = nn.Sequential(*encoder_layers)
#
#         self.to_latent = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(hidden_dim * seq_len * L * F, latent_dim)
#         )
#
#         self.from_latent = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim * seq_len * L * F),
#             nn.ReLU()
#         )
#
#         decoder_layers = []
#         in_channels = hidden_dim
#         for _ in range(num_blocks):
#             decoder_layers += [
#                 nn.ConvTranspose3d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm3d(hidden_dim),
#                 nn.ReLU()
#             ]
#         decoder_layers.append(nn.Conv3d(hidden_dim, 1, kernel_size=1))
#         self.decoder3d = nn.Sequential(*decoder_layers)
#
#     def forward(self, x):
#         B = x.size(0)
#         x = x.view(B, self.seq_len, self.L, self.F)  # [B, T, L, F]
#         x = x.unsqueeze(1)  # [B, 1, T, L, F]
#
#         feat = self.encoder3d(x)
#         z = self.to_latent(feat)
#
#         y = self.from_latent(z)
#         y = y.view(B, -1, self.seq_len, self.L, self.F)
#         recon = self.decoder3d(y)
#         recon = recon.squeeze(1).view(B, self.seq_len, -1)  # [B, T, D]
#         return recon, z