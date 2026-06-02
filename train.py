import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "dataset/BraTS2020_training_data/content/data"

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50


class BraTSDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def normalize(self, x):
        mn = x.min()
        mx = x.max()

        if mx > mn:
            x = (x - mn) / (mx - mn)

        return x.astype(np.float32)

    def __getitem__(self, idx):

        file_path = self.files[idx]

        with h5py.File(file_path, "r") as f:
            img = f["image"][()].astype(np.float32)

        t1 = img[:, :, 0]
        t1ce = img[:, :, 1]

        t1 = self.normalize(t1)
        t1ce = self.normalize(t1ce)

        t1 = np.expand_dims(t1, 0)
        t1ce = np.expand_dims(t1ce, 0)

        return (
            torch.tensor(t1),
            torch.tensor(t1ce)
        )


class Block(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.GroupNorm(1, c2),
            nn.GELU(),

            nn.Conv2d(c2, c2, 3, padding=1),
            nn.GroupNorm(1, c2),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = Block(1, 32)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = Block(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = Block(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.mid = Block(128, 128)

        self.u3 = Block(128 + 128, 128)
        self.u2 = Block(128 + 64, 64)
        self.u1 = Block(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        d1 = self.d1(x)

        d2 = self.d2(
            self.p1(d1)
        )

        d3 = self.d3(
            self.p2(d2)
        )

        m = self.mid(
            self.p3(d3)
        )

        u3 = F.interpolate(
            m,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        u3 = self.u3(
            torch.cat([u3, d3], 1)
        )

        u2 = F.interpolate(
            u3,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        u2 = self.u2(
            torch.cat([u2, d2], 1)
        )

        u1 = F.interpolate(
            u2,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        u1 = self.u1(
            torch.cat([u1, d1], 1)
        )

        return self.out(u1)


all_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".h5")
]

train_files, val_files = train_test_split(
    all_files,
    test_size=0.1,
    random_state=42
)

train_ds = BraTSDataset(train_files)
val_ds = BraTSDataset(val_files)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = UNet().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_loss = 1e9

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    loop = tqdm(train_loader)

    for x, y in loop:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = criterion(pred, y)

            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(
        f"Train={train_loss:.6f} "
        f"Val={val_loss:.6f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "t1_to_t1ce_model.pth"
        )

        print("Saved Best Model")
