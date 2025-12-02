import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter

# ================== 配置参数 ==================
CFG = {
    'context_length': 50,
    'future_length': 10,
    'D': 750,
    'n_timesteps': 2000,
    'lr': 1e-4,
    'batch_size': 128,
    'ae_epochs_context': 300,
    'ae_epochs_future': 200,
    'diff_epochs': 1000,
    'latent_dim': 16,
    'smooth_window': 51,
    'smooth_polyorder': 2
}


# ================== 多频段预处理 ==================
class MinMaxScalerCustom:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.scale = None
        self.min_range, self.max_range = feature_range

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.scale = self.max - self.min
        self.scale[self.scale == 0] = 1e-8

    def transform(self, data):
        norm = (data - self.min) / self.scale
        return norm * (self.max_range - self.min_range) + self.min_range

    def inverse_transform(self, data):
        norm = (data - self.min_range) / (self.max_range - self.min_range)
        return norm * self.scale + self.min


def preprocess_multi_frequency(df_multi):
    scaler = MinMaxScalerCustom()
    scaler.fit(df_multi)
    norm = scaler.transform(df_multi)
    smooth = savgol_filter(
        norm,
        window_length=CFG['smooth_window'],
        polyorder=CFG['smooth_polyorder'],
        axis=0, mode='interp'
    )
    return norm, smooth, scaler


# ================== 数据集定义 ==================
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiBandDataset(Dataset):
    def __init__(self, y_array, context_length, future_length, mask_ratio=0.1):
        self.context_length = context_length
        self.future_length = future_length
        self.mask_ratio = mask_ratio
        T, D = y_array.shape
        self.samples = [
            (y_array[i:i+context_length],
             y_array[i+context_length:i+context_length+future_length])
            for i in range(T - context_length - future_length)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, fut = self.samples[idx]
        ctx = torch.from_numpy(ctx).float()
        fut = torch.from_numpy(fut).float()

        # ========== 在 context 中屏蔽 25% ==========
        mask = torch.rand_like(ctx) < self.mask_ratio  # 25% True
        ctx = ctx.masked_fill(mask, 0.0)

        return ctx, fut
