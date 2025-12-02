import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from Context2CondNew import ContextTransformerAE           # ↖ 你的 Context AE
from F2Cconv1d import FutureAutoencoder               # ↖ 未来序列 AE
from DataSetPrepare import MultiBandDataset, preprocess_multi_frequency
from NoiseNet import DiffusionModel, inference           # ↖ Diffusion + 推理函数


# =============== 超参数配置 ===============
CFG = {
    'context_length': 50,
    'future_length': 10,
    'D': 750,
    'n_timesteps':1000,
    'lr': 1e-4,
    'batch_size': 256,
    'ae_epochs_context': 300,
    'ae_epochs_future': 300,
    'diff_epochs': 2000,
    'latent_dim': 32,
    'smooth_window': 61,
    'smooth_polyorder': 2,
}

CHECKPOINT_DIR = 'checkpoints'        # 保存权重目录
RESULT_DIR     = 'results'            # 图片保存目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR,     exist_ok=True)

# -----------------------------------------------------------
#  Dataset & Dataloader helpers
# -----------------------------------------------------------

def get_dataloaders(csv_path: str):
    """读 CSV → 归一化 → Dataset → Dataloader"""
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)[:, :CFG['D']]
    _, smooth, scaler = preprocess_multi_frequency(data)
    dataset = MultiBandDataset(smooth, CFG['context_length'], CFG['future_length'])

    N = len(dataset)
    split = int(N * 0.8)
    train_loader = DataLoader(Subset(dataset, list(range(split))), batch_size=CFG['batch_size'], shuffle=True)
    test_loader  = DataLoader(Subset(dataset, list(range(int(N * 0.555), int(N)))), batch_size=CFG['batch_size'], shuffle=False)
    return train_loader, test_loader, scaler

# -----------------------------------------------------------
#  Training helpers
# -----------------------------------------------------------

def train_autoencoder(model, loader, device, epochs):
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=CFG['lr'])
    model.train()
    for ep in range(epochs):
        total = 0.0
        for ctx, fut in loader:
            inp = ctx if isinstance(model, ContextTransformerAE) else fut
            inp = inp.to(device)
            recon, _ = model(inp)
            loss = F.mse_loss(recon, inp)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total += loss.item()
        print(f"AE [{model.__class__.__name__}] Epoch {ep+1}/{epochs}  loss={total/len(loader):.6f}")
    return model

def train_diffusion(ae_ctx, ae_fut, loader, device):
    model = DiffusionModel(cond_dim=CFG['latent_dim'], latent_dim=CFG['latent_dim'],
                           n_timestep=CFG['n_timesteps'], device=device).to(device)
    optimiser = optim.Adam(model.parameters(), lr=CFG['lr'])
    ae_ctx.eval()
    ae_fut.eval()
    for ep in range(CFG['diff_epochs']):
        total = 0.0
        for ctx, fut in loader:
            ctx, fut = ctx.to(device), fut.to(device)
            _, z_ctx = ae_ctx(ctx)    # (B, latent)
            _, z_fut = ae_fut(fut)    # (B, latent)
            B = z_fut.size(0)
            t = torch.randint(0, CFG['n_timesteps'], (B,), device=device)
            noise = torch.randn_like(z_fut)
            z_t = model.q_sample(z_fut, t, noise)
            noise_pred = model(z_t, z_ctx, t)
            loss = F.mse_loss(noise_pred, noise)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
        print(f"Diffusion Epoch {ep+1}/{CFG['diff_epochs']}  loss={total/len(loader):.6f}")
    return model


# -----------------------------------------------------------
#  Save / Load helpers
# -----------------------------------------------------------
def save_weights(ae_ctx, ae_fut, diffusion):
    torch.save(ae_ctx.state_dict(),  os.path.join(CHECKPOINT_DIR, 'ae_ctx.pth'))
    torch.save(ae_fut.state_dict(),  os.path.join(CHECKPOINT_DIR, 'ae_fut.pth'))
# def save_weights(diffusion):
    torch.save(diffusion.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion.pth'))


def load_weights(device):
    ae_ctx = ContextTransformerAE(seq_len=CFG['context_length'], input_dim=CFG['D'],
                                  latent_dim=CFG['latent_dim']).to(device)
    ae_fut = FutureAutoencoder(CFG['future_length'], CFG['D'], CFG['latent_dim']).to(device)
    diffusion = DiffusionModel(cond_dim=CFG['latent_dim'], latent_dim=CFG['latent_dim'],
                               n_timestep=CFG['n_timesteps'], device=device).to(device)
    ae_ctx.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'ae_ctx.pth'), map_location=device))
    ae_fut.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'ae_fut.pth'), map_location=device))
    diffusion.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion.pth'), map_location=device))
    ae_ctx.eval(); ae_fut.eval(); diffusion.eval()
    return ae_ctx, ae_fut, diffusion

# -----------------------------------------------------------
#  Visualization helpers
# -----------------------------------------------------------

def inverse_transform(mat, scaler):
    """mat: (N, D) — 使用 scaler 逆归一化"""
    mat_flat = scaler.inverse_transform(mat)      # 假设 scaler 实现了 inverse_transform
    return mat_flat

def visualise(test_loader, ae_ctx, ae_fut, diffusion, scaler, device):
    # 1) 收集所有样本的预测
    gt_list, pred_list = [], []
    with torch.no_grad():
        for ctx, fut in test_loader:
            pred = inference(ae_ctx, ae_fut, diffusion, ctx.to(device), device)   # (B, future_len, D)
            gt_list.append(fut[:, 0, :].cpu().numpy())
            pred_list.append(pred[:, 0, :])
    gt_mat   = np.concatenate(gt_list,   axis=0)   # (Nsamples, D)
    pred_mat = np.concatenate(pred_list, axis=0)

    # 平滑预测
    window1 = 6
    window2 = 3
    if len(pred_mat) >= window1:
        kernel1 = np.ones(window1) / window1
        kernel2 = np.ones(window2) / window2
        for i in range(pred_mat.shape[1]):
            pred_mat[:, i] = np.convolve(pred_mat[:, i], kernel1, mode='same')
            gt_mat[:, i] = np.convolve(gt_mat[:, i], kernel2, mode='same')

    # 逆归一化
    gt_mat   = inverse_transform(gt_mat,   scaler)
    pred_mat = inverse_transform(pred_mat, scaler)
    # 保存 Ground Truth
    gt_df = pd.DataFrame(gt_mat.reshape(gt_mat.shape[0], -1))  # 如果是三维，先展平为二维
    gt_df.to_csv('gt_mat.csv', index=False)

    # 保存 Prediction
    pred_df = pd.DataFrame(pred_mat.reshape(pred_mat.shape[0], -1))
    pred_df.to_csv('pred_matours.csv', index=False)


    # 2) 热力图
    all_vals = np.concatenate([gt_mat.ravel(), pred_mat.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im0 = axs[0].imshow(gt_mat.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('True Spectrogram'); axs[0].set_xlabel('Sample'); axs[0].set_ylabel('Freq Index')
    im1 = axs[1].imshow(pred_mat.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Predicted Spectrogram'); axs[1].set_xlabel('Sample')
    fig.colorbar(im0, ax=axs.ravel().tolist(), shrink=0.6, label='Power (dBm)')
    plt.tight_layout()
    heatmap_path = os.path.join(RESULT_DIR, 'spectrogram_comparison.png')
    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to {heatmap_path}")

    # 3) 选定频点曲线对比
    pick_bands = [0, 250, 500]
    fig2, axes = plt.subplots(len(pick_bands), 1, figsize=(14, 3 * len(pick_bands)), sharex=True)
    if len(pick_bands) == 1:
        axes = [axes]
    x_axis = np.arange(gt_mat.shape[0])
    for idx, band in enumerate(pick_bands):
        axes[idx].plot(x_axis, gt_mat[:, band],   label='GT')
        axes[idx].plot(x_axis, pred_mat[:, band], label='Pred', alpha=0.7)
        axes[idx].set_ylabel(f'Freq {band}')
        axes[idx].legend()
    axes[-1].set_xlabel('Test Sample Index')
    plt.tight_layout()
    curve_path = os.path.join(RESULT_DIR, 'band_curves.png')
    plt.savefig(curve_path, dpi=300)
    print(f"Band comparison curves saved to {curve_path}")
    plt.show()

# -----------------------------------------------------------
#  Main logic
# -----------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        print('\n>>> Training mode')
        train_loader, _, _ = get_dataloaders(args.csv)

        # 1. 训练两个自编码器
        # ae_ctx = ContextTransformerAE(seq_len=CFG['context_length'], input_dim=CFG['D'], latent_dim=CFG['latent_dim'])
        # ae_fut = FutureAutoencoder(CFG['future_length'], CFG['D'], CFG['latent_dim'])
        ae_ctx = ContextTransformerAE(seq_len=CFG['context_length'], input_dim=CFG['D'],
                                      latent_dim=CFG['latent_dim']).to(device)
        ae_fut = FutureAutoencoder(CFG['future_length'], CFG['D'], CFG['latent_dim']).to(device)
        # ae_ctx.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'ae_ctx.pth'), map_location=device))
        # ae_fut.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'ae_fut.pth'), map_location=device))
        # ae_ctx.eval()
        # ae_fut.eval()
        ae_ctx = train_autoencoder(ae_ctx, train_loader, device, CFG['ae_epochs_context'])
        ae_fut = train_autoencoder(ae_fut, train_loader, device, CFG['ae_epochs_future'])

        # 2. 训练扩散模型
        diffusion = train_diffusion(ae_ctx.to(device), ae_fut, train_loader, device)

        # 3. 保存权重
        save_weights(ae_ctx, ae_fut, diffusion)
        # save_weights(diffusion)
        print(f"Models saved to '{CHECKPOINT_DIR}' directory.")

    elif args.mode == 'test':
        print('\n>>> Test / Inference mode')
        # 加载数据 & 模型权重
        _, test_loader, scaler = get_dataloaders(args.csv)
        ae_ctx, ae_fut, diffusion = load_weights(device)
        # 可视化
        visualise(test_loader, ae_ctx, ae_fut, diffusion, scaler, device)
    else:
        raise ValueError("mode must be 'train' or 'test'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectrum Prediction • Train / Test pipeline')
    parser.add_argument('--csv',  type=str, default='./dataset/merged_power_data_sub6GHz_avg_per_minute.csv',
                        help='Path to CSV dataset file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train = fits model then saves; test = loads weights & visualises')
    args = parser.parse_args()
    main(args)