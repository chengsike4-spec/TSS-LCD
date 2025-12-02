# split_dataset.py
import argparse
from pathlib import Path

import pandas as pd


def split_csv(
    src_csv: str,
    train_csv: str,
    test_csv: str,
    train_ratio: float = 0.55,
    shuffle: bool = False,
    random_seed: int = 42,
):
    src_csv = Path(src_csv)
    df = pd.read_csv(src_csv)

    if shuffle:
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n = len(df)
    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(test_csv).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Total samples: {n}")
    print(f"Train samples: {len(train_df)} saved to {train_csv}")
    print(f"Test  samples: {len(test_df)} saved to {test_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split merged CSV into train/test CSVs")
    parser.add_argument(
        "--src",
        type=str,
        default="./dataset/merged_power_data_sub6GHz_avg_per_minute.csv",
        help="Path to merged CSV",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="./dataset/trainDataset.csv",
        help="Output path for train CSV",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./dataset/testDataset.csv",
        help="Output path for test CSV",
    )
    parser.add_argument("--ratio", type=float, default=0.55, help="Train ratio")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before split")
    args = parser.parse_args()

    split_csv(args.src, args.train, args.test, args.ratio, args.shuffle)
