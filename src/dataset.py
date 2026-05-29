"""
dataset.py
----------
PyTorch Dataset sınıfı — train/val/test CSV'lerini yükler ve
modelin beklediği tensor formatına dönüştürür.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROCESSED_DIR = os.path.join("data", "processed")
TARGET_COL    = "price"


class HousingDataset(Dataset):
    """
    Parameters
    ----------
    split : 'train' | 'val' | 'test'
        Hangi bölmenin yükleneceğini belirtir.
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), \
            "split parametresi 'train', 'val' veya 'test' olmalıdır."

        path = os.path.join(PROCESSED_DIR, f"{split}.csv")
        df   = pd.read_csv(path)

        self.feature_names = [c for c in df.columns if c != TARGET_COL]

        # Özellikler → float32 tensor
        self.X = torch.tensor(
            df[self.feature_names].values.astype(np.float32),
            dtype=torch.float32
        )

        # Hedef → float32 tensor, sütun vektörü (N, 1)
        self.y = torch.tensor(
            df[TARGET_COL].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_size(self):
        """Modelin giriş katmanı boyutu için kullanılır."""
        return self.X.shape[1]


# ─── Yardımcı fonksiyon ────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 32):
    """
    Üç split için DataLoader nesnelerini döndürür.

    Returns
    -------
    train_loader, val_loader, test_loader, input_size
    """
    train_ds = HousingDataset("train")
    val_ds   = HousingDataset("val")
    test_ds  = HousingDataset("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"[✓] DataLoader hazır  →  batch_size={batch_size}")
    print(f"    Train: {len(train_ds)} örnek  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")
    print(f"    Giriş boyutu: {train_ds.input_size}")

    return train_loader, val_loader, test_loader, train_ds.input_size


# ─── Hızlı test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, input_size = get_dataloaders(batch_size=32)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nÖrnek batch:")
    print(f"  X shape : {X_batch.shape}")   # (32, input_size)
    print(f"  y shape : {y_batch.shape}")   # (32, 1)
    print(f"  y örnek : {y_batch[:3].squeeze().tolist()}")
