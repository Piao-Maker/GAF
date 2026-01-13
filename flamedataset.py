import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from GAF import gaf_to_tensor

class FlameDatasetFrom3Dirs(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",          # "train" or "val"
        img_size=128,
        series_length=64,
        gaf_method="summation",
        use_augmentation=True,
    ):
        """
        root_dir: 数据集根目录，下面有 images/ labels/ time_series/
        split: "train" 或 "val"
        img_size: 图像和 GAF 的大小 (img_size x img_size)
        series_length: 时间序列重采样到的长度
        gaf_method: "summation" (GASF) 或 "difference" (GADF)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.series_length = series_length
        self.gaf_method = gaf_method

        self.images_dir = os.path.join(root_dir, "images", split)
        self.labels_dir = os.path.join(root_dir, "labels", split)
        self.ts_dir     = os.path.join(root_dir, "time_series", split)

        # 1. 找到所有图像文件
        img_paths = sorted(
            glob.glob(os.path.join(self.images_dir, "*.jpg")) +
            glob.glob(os.path.join(self.images_dir, "*.jpeg")) +
            glob.glob(os.path.join(self.images_dir, "*.png"))
        )
        assert len(img_paths) > 0, f"No images found in {self.images_dir}"

        self.samples = []
        for img_path in img_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # 假设 label 文件名为 basename + .txt / .npy / .csv 之一
            label_path = self._match_label_file(basename)
            ts_path = self._match_ts_file(basename)

            if label_path is None or ts_path is None:
                # 可以选择跳过，或者抛出异常
                # 这里选择跳过，并打印一下，你也可以改成 assert
                print(f"[WARN] Missing label or time_series for {basename}, skip.")
                continue

            self.samples.append({
                "img_path": img_path,
                "label_path": label_path,
                "ts_path": ts_path,
            })

        assert len(self.samples) > 0, "No valid samples found, please check your dataset structure."

        # 2. 图像预处理 / 数据增强
        if use_augmentation and split == "train":
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])

    def _match_label_file(self, basename):
        """
        假设 label 是独立文件，文件名和 image 一样，只是后缀不同。
        支持 .txt / .npy / .csv（你可以按需删减）
        """
        candidates = [
            os.path.join(self.labels_dir, basename + ".txt"),
            os.path.join(self.labels_dir, basename + ".npy"),
            os.path.join(self.labels_dir, basename + ".csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _match_ts_file(self, basename):
        """
        时间序列文件，假设一个样本对应一个 .npy 或 .csv 或 .txt
        """
        candidates = [
            os.path.join(self.ts_dir, basename + ".npy"),
            os.path.join(self.ts_dir, basename + ".csv"),
            os.path.join(self.ts_dir, basename + ".txt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)      # (3, H, W)
        return img

    def _load_label(self, label_path):
        """
        YOLO 检测标签 -> 图像级三分类标签：
        0: fire
        1: default
        2: smoke

        策略：
        - 有 fire(0) 框 → label = 0
        - 否则有 smoke(2) 框 → label = 2
        - 否则 → label = 1 (default)
        """
        # 若没有标签文件，直接当作 default
        if not os.path.exists(label_path):
            return torch.tensor(1, dtype=torch.long)

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 空文件 → default
        if len(lines) == 0:
            return torch.tensor(1, dtype=torch.long)

        has_fire = False
        has_smoke = False

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id == 0:
                has_fire = True
            elif class_id == 2:
                has_smoke = True

        if has_fire:
            label = 0  # fire
        elif has_smoke:
            label = 2  # smoke
        else:
            label = 1  # default

        return torch.tensor(label, dtype=torch.long)

    def _load_series(self, ts_path):
        """
        加载一维时间序列
        """
        if ts_path.endswith(".npy"):
            series = np.load(ts_path).astype(np.float32)
        else:
            # 默认 csv 或 txt，每行一个值
            series = np.loadtxt(ts_path, delimiter=",", dtype=np.float32)
        return series

    def _resample_series(self, series):
        """
        将时间序列重采样到固定长度 self.series_length
        """
        if len(series) == self.series_length:
            return series.astype(np.float32)

        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, self.series_length)
        series_resampled = np.interp(x_new, x_old, series)
        return series_resampled.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["img_path"]
        label_path = item["label_path"]
        ts_path = item["ts_path"]

        # 1. 图像
        img = self._load_image(img_path)           # (3, H, W)

        # 2. label
        label = self._load_label(label_path)       # scalar long

        # 3. 时间序列 -> 重采样 -> GAF
        series = self._load_series(ts_path)
        series = self._resample_series(series)
        gaf = gaf_to_tensor(series,
                            method=self.gaf_method,
                            image_size=self.img_size)  # (1, H, W)

        return img, gaf, label
