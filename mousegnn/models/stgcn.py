"""A lightweight ST-GCN implementation tailored for mouse behavior recognition."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GraphConv(nn.Module):
    """Spatial graph convolution using an adjacency matrix.

    Input shape: (B, C, T, V)
    Output shape: (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, bias: bool = True):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # propagate along spatial edges: B, C, T, V
        x = torch.einsum("vu,bctu->bctv", self.adjacency, x)
        return self.conv(x)


class TemporalConv(nn.Module):
    """Temporal convolution with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, stride: int = 1, residual: bool = True):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tcn = TemporalConv(out_channels, stride=stride)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.bn(x)
        x = self.relu(x + res)
        x = self.tcn(x)
        return x


class STGCN(nn.Module):
    def __init__(self, num_points: int, num_classes: int, in_channels: int = 2, dropout: float = 0.25):
        super().__init__()
        # chain graph if user does not provide custom adjacency
        A = torch.eye(num_points)
        for i in range(num_points - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(num_points * in_channels)
        self.layer1 = STGCNBlock(in_channels, 64, A, residual=False)
        self.layer2 = STGCNBlock(64, 64, A)
        self.layer3 = STGCNBlock(64, 64, A)
        self.layer4 = STGCNBlock(64, 128, A, stride=2)
        self.layer5 = STGCNBlock(128, 128, A)
        self.layer6 = STGCNBlock(128, 256, A, stride=2)
        self.layer7 = STGCNBlock(256, 256, A)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool(x).view(N, -1)
        x = self.dropout(x)
        return self.fc(x)


def create_model(num_points: int, num_classes: int, in_channels: int = 2, dropout: float = 0.25) -> STGCN:
    return STGCN(num_points=num_points, num_classes=num_classes, in_channels=in_channels, dropout=dropout)
