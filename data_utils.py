# data_utils.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DataSetPrepare import MultiBandDataset, preprocess_multi_frequency


def get_dataloader(
    csv_path: str,
    context_length: int,
    future_length: int,
    D: int,
    batch_size: int,
    shuffle: bool = True,
    return_scaler: bool = False,
):
    """
    读 CSV -> 归一化/平滑 -> MultiBandDataset -> DataLoader

    csv_path      : 训练或测试的 CSV 路径
    context_length: 历史长度
    future_length : 未来长度
    D             : 频点数
    batch_size    : 批大小
    shuffle       : 是否打乱
    return_scaler : 是否返回 scaler (测试时需要做 inverse_transform)
    """
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)[:, :D]  # (T, D)

    _, smooth, scaler = preprocess_multi_frequency(data)
    dataset = MultiBandDataset(smooth, context_length, future_length)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if return_scaler:
        return loader, scaler
    else:
        return loader
