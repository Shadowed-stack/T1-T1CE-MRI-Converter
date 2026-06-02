import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

from scipy.stats import pearsonr


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FILE = "sample.h5"


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

        self.u3 = Block(256, 128)
        self.u2 = Block(192, 64)
        self.u1 = Block(96, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))

        m = self.mid(self.p3(d3))

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


def norm(x):

    mn = x.min()
    mx = x.max()

    if mx > mn:
        x = (x - mn) / (mx - mn)

    return x


model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "t1_to_t1ce_model.pth",
        map_location=DEVICE
    )
)

model.eval()

with h5py.File(TEST_FILE, "r") as f:
    img = f["image"][()].astype(np.float32)

t1 = norm(img[:, :, 0])
gt = norm(img[:, :, 1])

x = torch.tensor(
    t1[None, None],
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():
    pred = model(x)

pred = pred.squeeze().cpu().numpy()
pred = norm(pred)

mse = mean_squared_error(gt, pred)
psnr = peak_signal_noise_ratio(gt, pred)
ssim = structural_similarity(gt, pred, data_range=1.0)

corr, _ = pearsonr(
    gt.flatten(),
    pred.flatten()
)

print("MSE:", mse)
print("PSNR:", psnr)
print("SSIM:", ssim)
print("Pearson:", corr)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(t1, cmap="gray")
plt.title("Input T1")

plt.subplot(1,3,2)
plt.imshow(gt, cmap="gray")
plt.title("Real T1CE")

plt.subplot(1,3,3)
plt.imshow(pred, cmap="gray")
plt.title("Generated T1CE")

plt.savefig(
    "comparison_output.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
