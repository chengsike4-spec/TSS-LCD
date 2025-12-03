# train.py
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from Context2CondNew import ContextTransformerAE
from F2Cnet import FutureAutoencoder
from NoiseNet import DiffusionModel
from data_utils import get_dataloader

from config import get_train_parser


# -----------------------------------------------------------
#  Training helpers
# -----------------------------------------------------------
def train_autoencoder(model, loader, device, epochs, lr):
    """
    通用 AE 训练函数：
    - context AE: 输入 ctx
    - future AE:  输入 fut
    """
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for ctx, fut in loader:
            # context AE 用 ctx，future AE 用 fut
            inp = ctx if isinstance(model, ContextTransformerAE) else fut
            inp = inp.to(device)

            recon, _ = model(inp)
            loss = F.mse_loss(recon, inp)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total += loss.item()
        print(
            f"AE [{model.__class__.__name__}] "
            f"Epoch {ep+1}/{epochs}  loss={total/len(loader):.6f}"
        )
    return model


def train_diffusion(ae_ctx, ae_fut, loader, device, args):
    """
    训练扩散模型：
    - 先用两个 AE 编码 ctx / fut
    - 再在 latent 空间上做 DDPM 训练
    """
    model = DiffusionModel(
        cond_dim=args.latent_dim,
        latent_dim=args.latent_dim,
        n_timestep=args.n_timesteps,
        device=device,
    ).to(device)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    ae_ctx.eval()
    ae_fut.eval()

    for ep in range(args.diff_epochs):
        total = 0.0
        for ctx, fut in loader:
            ctx, fut = ctx.to(device), fut.to(device)

            # 编码得到 latent
            _, z_ctx = ae_ctx(ctx)  # (B, latent_dim)
            _, z_fut = ae_fut(fut)  # (B, latent_dim)

            B = z_fut.size(0)
            t = torch.randint(0, args.n_timesteps, (B,), device=device)
            noise = torch.randn_like(z_fut)
            z_t = model.q_sample(z_fut, t, noise)

            noise_pred = model(z_t, z_ctx, t)
            loss = F.mse_loss(noise_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total += loss.item()
        print(
            f"Diffusion Epoch {ep+1}/{args.diff_epochs}  "
            f"loss={total/len(loader):.6f}"
        )
    return model


# -----------------------------------------------------------
#  Save helpers
# -----------------------------------------------------------
def save_weights(ae_ctx, ae_fut, diffusion, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(ae_ctx.state_dict(), os.path.join(checkpoint_dir, "ae_ctx.pth"))
    torch.save(ae_fut.state_dict(), os.path.join(checkpoint_dir, "ae_fut.pth"))
    torch.save(diffusion.state_dict(), os.path.join(checkpoint_dir, "diffusion.pth"))
    print(f"Models saved to '{checkpoint_dir}' directory.")


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(">>> Training mode")
    print(f"Using train CSV: {args.csv_train}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("Hyper-params  :")
    print(f"  context_length      = {args.context_length}")
    print(f"  future_length       = {args.future_length}")
    print(f"  D                   = {args.D}")
    print(f"  latent_dim          = {args.latent_dim}")
    print(f"  n_timesteps         = {args.n_timesteps}")
    print(f"  batch_size          = {args.batch_size}")
    print(f"  lr                  = {args.lr}")
    print(f"  ae_epochs_context   = {args.ae_epochs_context}")
    print(f"  ae_epochs_future    = {args.ae_epochs_future}")
    print(f"  diff_epochs         = {args.diff_epochs}")

    # dataloader
    train_loader = get_dataloader(
        csv_path=args.csv_train,
        context_length=args.context_length,
        future_length=args.future_length,
        D=args.D,
        batch_size=args.batch_size,
        shuffle=True,
        return_scaler=False,
    )

    # 1. 训练两个自编码器
    ae_ctx = ContextTransformerAE(
        seq_len=args.context_length,
        input_dim=args.D,
        latent_dim=args.latent_dim,
    ).to(device)

    ae_fut = FutureAutoencoder(
        args.future_length, args.D, args.latent_dim
    ).to(device)

    ae_ctx = train_autoencoder(
        ae_ctx,
        train_loader,
        device,
        epochs=args.ae_epochs_context,
        lr=args.lr,
    )
    ae_fut = train_autoencoder(
        ae_fut,
        train_loader,
        device,
        epochs=args.ae_epochs_future,
        lr=args.lr,
    )

    # 2. 训练扩散模型
    diffusion = train_diffusion(ae_ctx, ae_fut, train_loader, device, args)

    # 3. 保存权重
    save_weights(ae_ctx, ae_fut, diffusion, args.checkpoint_dir)


if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    main(args)
