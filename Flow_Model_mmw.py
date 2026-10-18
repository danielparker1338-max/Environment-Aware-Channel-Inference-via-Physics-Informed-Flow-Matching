import math

import torch
import torch.nn as nn


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.pool = nn.MaxPool2d(2) if downsample else None

    def forward(self, x, temb):
        residual = x
        x = x + self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        x = x + residual
        if self.pool is not None:
            x = self.pool(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, temb):
        residual = x
        x = x + self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class MiddleLayer(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, channels)

    def forward(self, x, temb):
        residual = x
        x = x + self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x + residual


class MiniUnet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.time_emb_dim = base_channels

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = nn.ModuleList(
            [
                DownLayer(base_channels, base_channels * 2, self.time_emb_dim),
                DownLayer(base_channels * 2, base_channels * 2, self.time_emb_dim),
            ]
        )
        self.maxpool1 = nn.MaxPool2d(2)

        self.down2 = nn.ModuleList(
            [
                DownLayer(base_channels * 2, base_channels * 4, self.time_emb_dim),
                DownLayer(base_channels * 4, base_channels * 4, self.time_emb_dim),
            ]
        )
        self.maxpool2 = nn.MaxPool2d(2)

        self.middle = MiddleLayer(base_channels * 4, self.time_emb_dim)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = nn.ModuleList(
            [
                UpLayer(base_channels * 8, base_channels * 2, self.time_emb_dim),
                UpLayer(base_channels * 2, base_channels * 2, self.time_emb_dim),
            ]
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = nn.ModuleList(
            [
                UpLayer(base_channels * 4, base_channels, self.time_emb_dim),
                UpLayer(base_channels, base_channels, self.time_emb_dim),
            ]
        )

        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def time_emb(self, t, dim):
        t = t * 1000.0
        half_dim = dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half_dim, device=t.device)
        )
        emb = torch.cat([torch.sin(t[:, None] / freqs), torch.cos(t[:, None] / freqs)], dim=-1)
        if emb.shape[1] < dim:
            emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
        return emb

    def forward(self, x, t):
        height, width = x.shape[-2:]
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(f"MiniUnet expects CSI height/width divisible by 4, got {height}x{width}.")

        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)

        for layer in self.down1:
            x = layer(x, temb)
        x1 = x
        x = self.maxpool1(x)

        for layer in self.down2:
            x = layer(x, temb)
        x2 = x
        x = self.maxpool2(x)

        x = self.middle(x, temb)

        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)

        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        return self.conv_out(x)
