import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {
    'context_length': 50,
    'future_length': 10,
    'D': 750,
    'n_timesteps': 20000,
    'lr': 1e-4,
    'batch_size': 128,
    'ae_epochs_context': 300,
    'ae_epochs_future': 300,
    'diff_epochs': 1000,
    'latent_dim': 16,
    'smooth_window': 21,
    'smooth_polyorder': 2,
}

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = np.linspace(0, T, steps)
    alpha_bar = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(beta, 1e-8, 0.999)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (-np.log(10000.0) / half_dim))
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class FiLMUNet1D(nn.Module):
    def __init__(self, in_dim, base_dim=64, out_dim=None, cond_dim=64):
        super().__init__()
        out_dim = out_dim or in_dim

        self.enc1 = nn.Conv1d(in_dim, base_dim, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(base_dim, base_dim * 2, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(base_dim * 2, base_dim * 4, kernel_size=3, padding=1)
        self.middle = nn.Conv1d(base_dim * 4, base_dim * 4, kernel_size=3, padding=1)
        self.dec3 = nn.Conv1d(base_dim * 4, base_dim * 2, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(base_dim * 4, base_dim, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(base_dim * 2, out_dim, kernel_size=3, padding=1)

        self.mod_enc1 = nn.Linear(cond_dim, base_dim)
        self.mod_enc2 = nn.Linear(cond_dim, base_dim * 2)
        self.mod_enc3 = nn.Linear(cond_dim, base_dim * 4)
        self.mod_mid  = nn.Linear(cond_dim, base_dim * 4)
        self.mod_dec3 = nn.Linear(cond_dim, base_dim * 2)
        self.mod_dec2 = nn.Linear(cond_dim, base_dim)
        self.mod_dec1 = nn.Linear(cond_dim, out_dim)

        self.relu = nn.ReLU()

    def modulate(self, h, emb, mod_layer):
        gamma = mod_layer(emb).unsqueeze(-1)
        return h * (1 + gamma)

    def forward(self, x, cond_emb):
        x = x.unsqueeze(-1)
        e1 = self.modulate(self.relu(self.enc1(x)), cond_emb, self.mod_enc1)
        e2 = self.modulate(self.relu(self.enc2(e1)), cond_emb, self.mod_enc2)
        e3 = self.modulate(self.relu(self.enc3(e2)), cond_emb, self.mod_enc3)

        m = self.modulate(self.relu(self.middle(e3)), cond_emb, self.mod_mid)

        d3 = self.modulate(self.relu(self.dec3(m)), cond_emb, self.mod_dec3)
        d2 = self.modulate(self.relu(self.dec2(torch.cat([d3, e2], dim=1))), cond_emb, self.mod_dec2)
        d1 = self.modulate(self.dec1(torch.cat([d2, e1], dim=1)), cond_emb, self.mod_dec1)

        return d1.squeeze(-1)

class DiffusionModel(nn.Module):
    def __init__(self, cond_dim, latent_dim, n_timestep, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_timestep = n_timestep
        self.device = device
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.time_embedding = SinusoidalTimeEmbedding(dim=32)
        self.noise_net = FiLMUNet1D(in_dim=latent_dim, out_dim=latent_dim, cond_dim=latent_dim + 32)

        betas = cosine_beta_schedule(n_timestep)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

    def q_sample(self, z0, t, noise):
        a_bar = self.alpha_cumprod[t].view(-1, 1)
        om = 1 - a_bar
        return torch.sqrt(a_bar) * z0 + torch.sqrt(om) * noise

    def forward(self, zt, cond_z, t):
        t_emb = self.time_embedding(t)  # [B, 32]
        cond_emb = torch.cat([self.cond_proj(cond_z), t_emb], dim=1)
        return self.noise_net(zt, cond_emb)

    def p_sample(self, zt, cond_z, t):
        beta_t = self.betas[t].view(-1, 1)
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alpha_cumprod[t].view(-1, 1)
        alpha_bar_tm1 = self.alpha_cumprod_prev[t].view(-1, 1)

        e_pred = self.forward(zt, cond_z, t)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_alpha_bar_tm1 = torch.sqrt(alpha_bar_tm1)
        denom = 1 - alpha_bar_t

        A_t = (1 / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom) + (sqrt_alpha_t * (1 - alpha_bar_tm1) / denom)
        B_t = (torch.sqrt(1 - alpha_bar_t) / sqrt_alpha_t) * (sqrt_alpha_bar_tm1 * beta_t / denom)
        sigma_t = torch.sqrt((1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t)

        noise = torch.randn_like(zt)
        mask = (t > 0).float().view(-1, 1)
        return A_t * zt - B_t * e_pred + mask * sigma_t * noise

    def p_sample_loop(self, cond_z):
        B = cond_z.size(0)
        z = torch.randn(B, self.latent_dim, device=self.device)
        for step in reversed(range(self.n_timestep)):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            z = self.p_sample(z, cond_z, t)
        return z

def inference(ae_ctx, ae_fut, diffusion, ctx, device):
    ae_ctx.eval()
    ae_fut.eval()
    diffusion.eval()
    with torch.no_grad():
        _, z_ctx = ae_ctx(ctx.to(device))
        z_sample = diffusion.p_sample_loop(z_ctx)
        recon_flat = ae_fut.decoder(z_sample)
        out = recon_flat.view(-1, CFG['future_length'], CFG['D'])
    return out.cpu().numpy()