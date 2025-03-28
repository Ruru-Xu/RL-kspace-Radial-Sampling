import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fftn, ifftn, fftshift, ifftshift

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm architecture for better stability
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        B, C, H, W = x.size()

        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(y).view(B, -1, H * W)
        value = self.value_conv(y).view(B, -1, H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return x + self.gamma * out

class Kspace_Net_MT(nn.Module):

    def __init__(self, act_dim, feature_dim, mt_shape, dropout=0.0):
        super().__init__()
        self.act_dim = act_dim
        self.mt_shape = mt_shape
        patch_size, num_heads, num_layers = 16, 8, 6  # Increased heads and layers

        # Enhanced k-space processing
        self.kspace_conv = nn.Sequential(
            ResidualBlock(2, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2)
        )

        # Enhanced image processing
        self.image_conv = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2)
        )

        # Enhanced fusion
        self.fusion = CrossAttentionFusion(32)

        # Enhanced transformer
        self.patch_embed = nn.Conv2d(32, feature_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = TransformerEncoder(
            embed_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Enhanced trunk
        self.trunk = nn.Sequential(
            nn.Linear(16384, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        # Enhanced policy head
        self.policy_layer = nn.Sequential(
            nn.Linear(feature_dim + mt_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.act_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input_dict):
        kspace = input_dict['kspace'] + 1e-6 #[16, 1, 256, 256] complex64
        mt = input_dict['mt'] #tensor(73, device='cuda:0')
        mt_vec = F.one_hot(mt, num_classes=self.mt_shape[0]).float() #(180,)

        kspace_combined = torch.cat([kspace.real, kspace.imag], dim=1)
        kspace_features = self.kspace_conv(kspace_combined)

        image = ifftshift(ifftn(fftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)).abs()
        image_features = self.image_conv(image)

        # Enhanced fusion
        combined_features = self.fusion(kspace_features, image_features)

        # Rest of the forward pass remains the same
        patches = self.patch_embed(combined_features)
        batch, embed_dim, num_patches_h, num_patches_w = patches.shape
        transformer_input = patches.view(batch, embed_dim, -1).permute(0, 2, 1)

        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.permute(0, 2, 1).view(batch, embed_dim, num_patches_h, num_patches_w)

        pooled_features = self.pool(transformer_output)
        features_flat = pooled_features.reshape(pooled_features.size(0), -1)

        h = self.trunk(features_flat)

        if len(mt_vec.shape) == 1:
            h_combined = torch.cat((h, mt_vec.repeat(h.shape[0], 1)), dim=-1)
        else:
            h_combined = torch.cat((h, mt_vec), dim=-1)

        return self.policy_layer(h_combined)

class Kspace_Net_Critic_MT(nn.Module):

    def __init__(self, feature_dim, mt_shape, dropout=0.0):
        super().__init__()
        self.mt_shape = mt_shape
        patch_size, num_heads, num_layers = 16, 8, 6  # Increased heads and layers

        # Enhanced k-space processing
        self.kspace_conv = nn.Sequential(
            ResidualBlock(2, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2)
        )

        # Enhanced image processing
        self.image_conv = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2)
        )

        # Enhanced fusion
        self.fusion = CrossAttentionFusion(32)

        # Enhanced transformer
        self.patch_embed = nn.Conv2d(32, feature_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = TransformerEncoder(
            embed_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Enhanced trunk
        self.trunk = nn.Sequential(
            nn.Linear(16384, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        # Critic layer
        self.critic_layer = nn.Sequential(
            nn.Linear(feature_dim + mt_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


    def forward(self, input_dict):
        kspace = input_dict['kspace'] + 1e-6
        mt = input_dict['mt']
        mt_vec = F.one_hot(mt, num_classes=self.mt_shape[0]).float()

        kspace_combined = torch.cat([kspace.real, kspace.imag], dim=1)
        kspace_features = self.kspace_conv(kspace_combined)

        image = ifftshift(ifftn(fftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)).abs()
        image_features = self.image_conv(image)

        # Enhanced fusion
        combined_features = self.fusion(kspace_features, image_features)

        # Rest of the forward pass remains the same
        patches = self.patch_embed(combined_features)
        batch, embed_dim, num_patches_h, num_patches_w = patches.shape
        transformer_input = patches.view(batch, embed_dim, -1).permute(0, 2, 1)

        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.permute(0, 2, 1).view(batch, embed_dim, num_patches_h, num_patches_w)

        pooled_features = self.pool(transformer_output)
        features_flat = pooled_features.reshape(pooled_features.size(0), -1)

        h = self.trunk(features_flat)

        if len(mt_vec.shape) == 1:
            h_combined = torch.cat((h, mt_vec.repeat(h.shape[0], 1)), dim=-1)
        else:
            h_combined = torch.cat((h, mt_vec), dim=-1)

        # Compute value estimate
        value = self.critic_layer(h_combined).squeeze()

        return value


