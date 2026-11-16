"""Dataset and multimodal bottleneck transformer for the sensor classifier."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SENSOR_DIMS = (12, 13, 9, 12, 8)


class SensorFusionDataset(Dataset):
    """Split standardized sensor features into modality tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if features.ndim != 2 or features.shape[1] != sum(SENSOR_DIMS):
            raise ValueError(
                f"Expected features with {sum(SENSOR_DIMS)} columns; "
                f"received shape {features.shape}"
            )
        if len(features) != len(labels):
            raise ValueError("Feature and label counts do not match")

        self.sensor_features = []
        start = 0
        for width in SENSOR_DIMS:
            end = start + width
            self.sensor_features.append(
                torch.as_tensor(features[:, start:end], dtype=torch.float32)
            )
            start = end
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return tuple(sensor[index] for sensor in self.sensor_features) + (
            self.labels[index],
        )


class MultiSensorBottleneckTransformer(nn.Module):
    """Fuse one projected token per sensor through learned bottleneck tokens."""

    def __init__(
        self,
        sensor_dims=SENSOR_DIMS,
        d_model: int = 128,
        num_bottlenecks: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(sensor_dims) != len(SENSOR_DIMS):
            raise ValueError(f"Expected {len(SENSOR_DIMS)} sensor dimensions")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        self.sensor_projections = nn.ModuleList(
            nn.Sequential(nn.Linear(width, d_model), nn.LayerNorm(d_model))
            for width in sensor_dims
        )
        self.sensor_embeddings = nn.Parameter(
            torch.empty(1, len(sensor_dims), d_model)
        )
        self.bottleneck_tokens = nn.Parameter(
            torch.empty(1, num_bottlenecks, d_model)
        )
        nn.init.normal_(self.sensor_embeddings, std=0.02)
        nn.init.normal_(self.bottleneck_tokens, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, *sensor_inputs: torch.Tensor) -> torch.Tensor:
        if len(sensor_inputs) != len(self.sensor_projections):
            raise ValueError(
                f"Expected {len(self.sensor_projections)} sensor inputs; "
                f"received {len(sensor_inputs)}"
            )

        sensor_tokens = torch.stack(
            [
                projection(sensor) + self.sensor_embeddings[:, index]
                for index, (projection, sensor) in enumerate(
                    zip(self.sensor_projections, sensor_inputs)
                )
            ],
            dim=1,
        )
        bottlenecks = self.bottleneck_tokens.expand(sensor_tokens.size(0), -1, -1)
        fused = self.fusion(torch.cat((sensor_tokens, bottlenecks), dim=1))
        fused_bottlenecks = fused[:, sensor_tokens.size(1) :, :]
        return self.classifier(fused_bottlenecks.mean(dim=1))
