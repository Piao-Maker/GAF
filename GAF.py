import numpy as np
import torch
import torch.nn.functional as F


def min_max_scale(series):
    """把一维时间序列缩放到 [-1, 1]"""
    x = np.asarray(series, dtype=np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        # 避免除以 0：如果全是一个值，就直接返回全 0
        return np.zeros_like(x)
    x_norm = (x - x_min) / (x_max - x_min)      # [0, 1]
    x_scaled = x_norm * 2 - 1                   # [-1, 1]
    return x_scaled


def gaf_transform_numpy(series, method="summation"):
    """
    利用 Gramian 角场将一维时间序列转换为二维矩阵 (n x n).
    method: "summation" -> GASF, "difference" -> GADF
    """
    x_scaled = min_max_scale(series)            # [-1, 1]
    phi = np.arccos(x_scaled)                  # 角度 [0, π], shape (n,)

    phi_i = phi.reshape(-1, 1)                 # (n, 1)
    phi_j = phi.reshape(1, -1)                 # (1, n)

    if method == "summation":                  # GASF
        gaf = np.cos(phi_i + phi_j)
    elif method == "difference":               # GADF
        gaf = np.sin(phi_i - phi_j)
    else:
        raise ValueError("method must be 'summation' or 'difference'")
    return gaf.astype(np.float32)


def gaf_to_tensor(series, method="summation", image_size=None):
    """
    返回 shape = (1, H, W) 的 torch.Tensor，单通道 GAF 图像
    """
    if isinstance(series, torch.Tensor):
        series = series.detach().cpu().numpy()

    gaf = gaf_transform_numpy(series, method=method)      # (n, n)
    gaf_tensor = torch.from_numpy(gaf).unsqueeze(0)       # (1, H, W)

    if image_size is not None:
        gaf_tensor = F.interpolate(
            gaf_tensor.unsqueeze(0),                      # (1, 1, H, W)
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)                                      # (1, image_size, image_size)

    return gaf_tensor
