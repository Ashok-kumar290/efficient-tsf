"""Real benchmark data: ETTh1 with the STANDARD LTSF protocol.

This is the protocol every LTSF paper uses, so our numbers are comparable:
  - 7 variables (hourly electricity-transformer sensors).
  - Splits (Informer convention): train = first 12 months, val = next 4, test = next 4.
  - Normalization fit on TRAIN ONLY, applied to all splits (no leakage).
  - Sliding windows: seq_len history -> pred_len forecast.
Auto-downloads ETTh1.csv from the public ETDataset repo on first use.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def download_etth1(path: str = "data/ETTh1.csv") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading ETTh1 -> {path}")
        urllib.request.urlretrieve(ETTH1_URL, path)
    return path


class ETTh1(Dataset):
    N_VARS = 7

    def __init__(self, split: str = "train", seq_len: int = 96, pred_len: int = 96,
                 path: str | None = None):
        assert split in {"train", "val", "test"}
        df = pd.read_csv(path or download_etth1())
        data = df[[c for c in df.columns if c != "date"]].values.astype(np.float32)

        m, v = 12 * 30 * 24, 4 * 30 * 24                 # 12-month, 4-month spans (hourly)
        lo = {"train": 0, "val": m - seq_len, "test": m + v - seq_len}
        hi = {"train": m, "val": m + v, "test": m + 2 * v}

        train = data[0:m]
        self.mean, self.std = train.mean(0), train.std(0) + 1e-8
        data = (data - self.mean) / self.std             # train-only normalization
        self.data = data[lo[split]:hi[split]]
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, i: int):
        x = self.data[i:i + self.seq_len]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)
