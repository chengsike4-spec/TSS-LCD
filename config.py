# config.py
import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    把模型结构 & 训练/测试都用得到的公共超参数挂到 parser 上。
    所有脚本都通过 args.xxx 访问这些参数。
    """
    parser.add_argument(
        "--context-length",
        type=int,
        default=50,
        help="Context sequence length (T1)",
    )
    parser.add_argument(
        "--future-length",
        type=int,
        default=10,
        help="Future sequence length (T2)",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=750,
        help="Feature dimension (e.g., #freq bins)",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for all optimizers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension",
    )
    parser.add_argument(
        "--ae-epochs-context",
        type=int,
        default=300,
        help="Epochs for context autoencoder",
    )
    parser.add_argument(
        "--ae-epochs-future",
        type=int,
        default=300,
        help="Epochs for future autoencoder",
    )
    parser.add_argument(
        "--diff-epochs",
        type=int,
        default=2000,
        help="Epochs for diffusion model",
    )
    return parser


def get_train_parser() -> argparse.ArgumentParser:
    """
    训练脚本专用 parser：
    - 先挂公共超参数
    - 再挂 train 特有参数
    """
    parser = argparse.ArgumentParser(description="Spectrum Prediction • Train only")
    add_common_args(parser)
    parser.add_argument(
        "--csv-train",
        type=str,
        default="./dataset/trainDataset.csv",
        help="Path to train CSV file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model weights (.pth files)",
    )
    return parser


def get_test_parser() -> argparse.ArgumentParser:
    """
    测试脚本专用 parser：
    - 先挂公共超参数（需要保证和训练时一致）
    - 再挂 test 特有参数
    """
    parser = argparse.ArgumentParser(description="Spectrum Prediction • Test only")
    add_common_args(parser)
    parser.add_argument(
        "--csv-test",
        type=str,
        default="./dataset/testDataset.csv",
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing model weights (.pth files)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Directory to save visualisation outputs",
    )
    return parser
