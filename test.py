# test.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from Context2CondNew import ContextTransformerAE
from F2Cnet import FutureAutoencoder
from NoiseNet import DiffusionModel, inference
from data_utils import get_dataloader

from config import get_test_parser


# -----------------------------------------------------------
#  Load helpers
# -----------------------------------------------------------
def load_weights(device, args):
    """
    从 args.checkpoint_dir 加载三个子网络的权重。
    所有结构超参数直接用 args.xxx。
    """
    ae_ctx = ContextTransformerAE(
        seq_len=args.context_length,
        input_dim=args.D,
        latent_dim=args.latent_dim,
    ).to(device)

    ae_fut = FutureAutoencoder(
        args.future_length, args.D, args.latent_dim
    ).to(device)

    diffusion = DiffusionModel(
        cond_dim=args.latent_dim,
        latent_dim=args.latent_dim,
        n_timestep=args.n_timesteps,
        device=device,
    ).to(device)

    ckpt_dir = args.checkpoint_dir

    ae_ctx.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ae_ctx.pth"), map_location=device)
    )
    ae_fut.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "ae_fut.pth"), map_location=device)
    )
    diffusion.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "diffusion.pth"), map_location=device)
    )

    ae_ctx.eval()
    ae_fut.eval()
    diffusion.eval()
    print(f"Loaded weights from '{ckpt_dir}'")
    return ae_ctx, ae_fut, diffusion


# -----------------------------------------------------------
#  Visualization helpers
# -----------------------------------------------------------
def inverse_transform(mat, scaler):
    """
    mat: (N, D) — 使用 scaler 逆归一化
    """
    return scaler.inverse_transform(mat)


def visualise(test_loader, ae_ctx, ae_fut, diffusion, scaler, device, args):
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    # 1) 收集所有样本的预测
    gt_list, pred_list = [], []
    with torch.no_grad():
        for ctx, fut in test_loader:
            ctx = ctx.to(device)
            pred = inference(ae_ctx, ae_fut, diffusion, ctx, device)  # (B, future_len, D)

            gt_list.append(fut[:, 0, :].cpu().numpy())
            pred_list.append(pred[:, 0, :])

    gt_mat = np.concatenate(gt_list, axis=0)   # (Nsamples, D)
    pred_mat = np.concatenate(pred_list, axis=0)

    # 简单平滑（如需可加到 parser 里）
    window1 = 1
    window2 = 1
    if len(pred_mat) >= window1:
        kernel1 = np.ones(window1) / window1
        kernel2 = np.ones(window2) / window2
        for i in range(pred_mat.shape[1]):
            pred_mat[:, i] = np.convolve(pred_mat[:, i], kernel1, mode="same")
            gt_mat[:, i] = np.convolve(gt_mat[:, i], kernel2, mode="same")

    # 逆归一化
    gt_mat = inverse_transform(gt_mat, scaler)
    pred_mat = inverse_transform(pred_mat, scaler)

    # 保存 Ground Truth
    gt_df = pd.DataFrame(gt_mat.reshape(gt_mat.shape[0], -1))
    gt_df.to_csv(os.path.join(result_dir, "gt_mat.csv"), index=False)

    # 保存 Prediction
    pred_df = pd.DataFrame(pred_mat.reshape(pred_mat.shape[0], -1))
    pred_df.to_csv(os.path.join(result_dir, "pred_matours.csv"), index=False)

    # 2) 热力图
    all_vals = np.concatenate([gt_mat.ravel(), pred_mat.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im0 = axs[0].imshow(
        gt_mat.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axs[0].set_title("True Spectrogram")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Freq Index")

    im1 = axs[1].imshow(
        pred_mat.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axs[1].set_title("Predicted Spectrogram")
    axs[1].set_xlabel("Sample")

    fig.colorbar(im0, ax=axs.ravel().tolist(), shrink=0.6, label="Power (dBm)")
    plt.tight_layout()
    heatmap_path = os.path.join(result_dir, "spectrogram_comparison.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to {heatmap_path}")

    # 3) 选定频点曲线对比
    pick_bands = [100, 350, 500]
    fig2, axes = plt.subplots(len(pick_bands), 1, figsize=(14, 3 * len(pick_bands)), sharex=True)
    if len(pick_bands) == 1:
        axes = [axes]
    x_axis = np.arange(gt_mat.shape[0])
    for idx, band in enumerate(pick_bands):
        axes[idx].plot(x_axis, gt_mat[:, band], label="GT")
        axes[idx].plot(x_axis, pred_mat[:, band], label="Pred", alpha=0.7)
        axes[idx].set_ylabel(f"Freq {band}")
        axes[idx].legend()
    axes[-1].set_xlabel("Test Sample Index")
    plt.tight_layout()
    curve_path = os.path.join(result_dir, "band_curves.png")
    plt.savefig(curve_path, dpi=300)
    print(f"Band comparison curves saved to {curve_path}")
    plt.show()


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Test / Inference mode")
    print(f"Using test CSV : {args.csv_test}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Result dir    : {args.result_dir}")
    print("Hyper-params  :")
    print(f"  context_length = {args.context_length}")
    print(f"  future_length  = {args.future_length}")
    print(f"  D              = {args.D}")
    print(f"  latent_dim     = {args.latent_dim}")
    print(f"  n_timesteps    = {args.n_timesteps}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  lr             = {args.lr}")

    # dataloader
    test_loader, scaler = get_dataloader(
        csv_path=args.csv_test,
        context_length=args.context_length,
        future_length=args.future_length,
        D=args.D,
        batch_size=args.batch_size,
        shuffle=False,
        return_scaler=True,
    )

    # 加载权重
    ae_ctx, ae_fut, diffusion = load_weights(device, args)

    # 可视化
    visualise(test_loader, ae_ctx, ae_fut, diffusion, scaler, device, args)


if __name__ == "__main__":
    parser = get_test_parser()
    args = parser.parse_args()
    main(args)
