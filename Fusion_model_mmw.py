import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VE_MultimodalityMMW(nn.Module):
    def __init__(
        self,
        channel_shape=(16, 64),
        output_channels=2,
        gps_embed_dim=128,
        attention_heads=4,
    ):
        super().__init__()
        self.channel_shape = tuple(channel_shape)
        self.output_channels = int(output_channels)
        self.gps_embed_dim = gps_embed_dim
        self.feature_dim = self.output_channels * self.channel_shape[0] * self.channel_shape[1]

        if self.feature_dim % attention_heads != 0:
            raise ValueError(
                f"feature_dim={self.feature_dim} must be divisible by attention_heads={attention_heads}."
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        hidden_dim = min(max(self.feature_dim, 512), 2048)

        self.gps_encoder = nn.Sequential(
            nn.Linear(gps_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.target_gps_encoder = nn.Sequential(
            nn.Linear(gps_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.ReLU(),
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, self.feature_dim),
            nn.ReLU(),
        )

        self.point_conv1 = nn.Conv1d(3, 128, kernel_size=1)
        self.point_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.point_conv3 = nn.Conv1d(128, 512, kernel_size=1)
        self.point_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.ReLU(),
        )

        self.fusion = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=attention_heads,
            batch_first=False,
        )
        self.fc_mu = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.fc_logvar = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)

    def pos_embedding(self, coords, embed_dim=None, scale=1000.0):
        embed_dim = embed_dim or self.gps_embed_dim
        if coords.ndim != 2:
            raise ValueError(f"GPS coordinates must have shape [B, D], got {tuple(coords.shape)}.")

        dims = coords.shape[1]
        pairs_per_dim = max(embed_dim // (2 * dims), 1)
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), pairs_per_dim, device=coords.device)
        )

        parts = []
        scaled = coords * scale
        for dim in range(dims):
            value = scaled[:, dim : dim + 1]
            parts.append(torch.sin(value / freqs))
            parts.append(torch.cos(value / freqs))

        emb = torch.cat(parts, dim=1)
        if emb.shape[1] < embed_dim:
            emb = F.pad(emb, (0, embed_dim - emb.shape[1]))
        return emb[:, :embed_dim]

    def encode_points(self, points):
        points = F.relu(self.point_conv1(points))
        points = F.relu(self.point_conv2(points))
        points = F.relu(self.point_conv3(points))
        points = torch.max(points, dim=-1)[0]
        return self.point_mlp(points)

    def _target_position_embedding(self, target_gps, link_geometry=None):
        if link_geometry is not None:
            target_gps = torch.cat([target_gps, link_geometry], dim=1)
        return self.pos_embedding(target_gps)

    def encode_joint(self, image, sensor_gps, target_gps, points, link_geometry=None):
        image_feature = self.image_encoder(image)
        sensor_gps_feature = self.gps_encoder(self.pos_embedding(sensor_gps))
        target_gps_feature = self.target_gps_encoder(
            self._target_position_embedding(target_gps, link_geometry)
        )
        point_feature = self.encode_points(points)

        stacked = torch.stack(
            [image_feature, sensor_gps_feature, target_gps_feature, point_feature],
            dim=0,
        )
        fused, _ = self.fusion(stacked, stacked, stacked)
        fused = F.relu(fused.mean(dim=0))
        fused = fused.view(-1, self.output_channels, self.channel_shape[0], self.channel_shape[1])
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar

    def encode_joint_dual(
        self,
        rsu_image,
        rsu_gps,
        rsu_points,
        cav_image,
        cav_gps,
        cav_points,
        target_gps,
        link_geometry=None,
    ):
        rsu_image_feature = self.image_encoder(rsu_image)
        cav_image_feature = self.image_encoder(cav_image)
        rsu_gps_feature = self.gps_encoder(self.pos_embedding(rsu_gps))
        cav_gps_feature = self.gps_encoder(self.pos_embedding(cav_gps))
        target_gps_feature = self.target_gps_encoder(
            self._target_position_embedding(target_gps, link_geometry)
        )
        rsu_point_feature = self.encode_points(rsu_points)
        cav_point_feature = self.encode_points(cav_points)

        stacked = torch.stack(
            [
                rsu_image_feature,
                rsu_gps_feature,
                rsu_point_feature,
                cav_image_feature,
                cav_gps_feature,
                cav_point_feature,
                target_gps_feature,
            ],
            dim=0,
        )
        fused, _ = self.fusion(stacked, stacked, stacked)
        fused = F.relu(fused.mean(dim=0))
        fused = fused.view(-1, self.output_channels, self.channel_shape[0], self.channel_shape[1])
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, sensor_gps, target_gps, points, link_geometry=None):
        mu, logvar = self.encode_joint(image, sensor_gps, target_gps, points, link_geometry)
        return self.reparameterize(mu, logvar), mu, logvar

    def forward_dual(
        self,
        rsu_image,
        rsu_gps,
        rsu_points,
        cav_image,
        cav_gps,
        cav_points,
        target_gps,
        link_geometry=None,
    ):
        mu, logvar = self.encode_joint_dual(
            rsu_image,
            rsu_gps,
            rsu_points,
            cav_image,
            cav_gps,
            cav_points,
            target_gps,
            link_geometry,
        )
        return self.reparameterize(mu, logvar), mu, logvar
