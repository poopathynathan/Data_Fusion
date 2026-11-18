"""
Google-style specialized multi-bottleneck feature-fusion model.

Input:
    X: [batch, 54] standardized scalar features in FEATURE_NAMES order.

Architecture:
    54 scalar feature tokens -> 4 domain-guided bottleneck tokens.
    Each bottleneck attends only to its explicitly routed features.
    The four bottleneck tokens then communicate with one another.
    This process repeats for the configured number of fusion layers.

This is a new architecture. It does not overwrite the original MBT.
"""

from typing import Dict, List
import torch
import torch.nn as nn


FEATURE_NAMES = [
    "radar_slant_range_m",
    "radar_azimuth_deg",
    "radar_elevation_deg",
    "radar_radial_velocity_mps",
    "radar_radial_acceleration_mps2",
    "radar_trajectory_curvature",
    "radar_rcs_mean_dbsm",
    "radar_rcs_variance",
    "radar_micro_doppler_peak_hz",
    "radar_doppler_bandwidth_hz",
    "radar_snr_db",
    "radar_detection_confidence",
    "df_aoa_azimuth_deg",
    "df_aoa_elevation_deg",
    "df_bearing_rate_degps",
    "df_carrier_frequency_mhz",
    "df_bandwidth_mhz",
    "df_hopping_rate_hops_sec",
    "df_rssi_dbm",
    "df_snr_db",
    "df_mod_is_fhss",
    "df_mod_is_dsss",
    "df_mod_is_ofdm",
    "df_mod_is_noise_none",
    "df_pulse_repetition_interval_ms",
    "tv_bbox_center_x",
    "tv_bbox_center_y",
    "tv_bbox_width",
    "tv_bbox_height",
    "tv_aspect_ratio",
    "tv_normalized_pixel_area",
    "tv_optical_flow_vx",
    "tv_optical_flow_vy",
    "tv_detector_confidence",
    "it_bbox_center_x",
    "it_bbox_center_y",
    "it_bbox_width",
    "it_bbox_height",
    "it_aspect_ratio",
    "it_target_mean_temp_c",
    "it_background_temp_c",
    "it_temp_contrast_delta_t_c",
    "it_peak_intensity_norm",
    "it_thermal_spatial_variance",
    "it_hotspot_count",
    "it_detector_confidence",
    "pan_mount_azimuth_deg",
    "pan_mount_elevation_deg",
    "pan_mount_slew_rate_az_degps",
    "pan_mount_slew_rate_el_degps",
    "pan_background_clutter_index",
    "pan_ambient_light_level_norm",
    "pan_visibility_haze_index_norm",
    "pan_track_duration_sec",
]

# Feature-to-bottleneck routing.
# Indices are derived from FEATURE_NAMES above.
# A feature may be routed to more than one bottleneck.
ROUTING_BY_NAME: Dict[str, List[str]] = {
    # B1: kinematics and geometry
    "radar_slant_range_m": ["B1"],
    "radar_azimuth_deg": ["B1"],
    "radar_elevation_deg": ["B1"],
    "radar_radial_velocity_mps": ["B1"],
    "radar_radial_acceleration_mps2": ["B1"],
    "radar_trajectory_curvature": ["B1"],
    "df_aoa_azimuth_deg": ["B1"],
    "df_aoa_elevation_deg": ["B1"],
    "df_bearing_rate_degps": ["B1"],
    "tv_bbox_center_x": ["B1"],
    "tv_bbox_center_y": ["B1"],
    "tv_optical_flow_vx": ["B1"],
    "tv_optical_flow_vy": ["B1"],
    "it_bbox_center_x": ["B1"],
    "it_bbox_center_y": ["B1"],
    "pan_mount_azimuth_deg": ["B1"],
    "pan_mount_elevation_deg": ["B1"],
    "pan_mount_slew_rate_az_degps": ["B1"],
    "pan_mount_slew_rate_el_degps": ["B1"],

    # B2: RF and micro-Doppler
    "radar_rcs_mean_dbsm": ["B2"],
    "radar_rcs_variance": ["B2"],
    "radar_micro_doppler_peak_hz": ["B2"],
    "radar_doppler_bandwidth_hz": ["B2"],
    "radar_snr_db": ["B2", "B4"],
    "df_carrier_frequency_mhz": ["B2"],
    "df_bandwidth_mhz": ["B2"],
    "df_hopping_rate_hops_sec": ["B2"],
    "df_rssi_dbm": ["B2", "B4"],
    "df_snr_db": ["B2", "B4"],
    "df_mod_is_fhss": ["B2"],
    "df_mod_is_dsss": ["B2"],
    "df_mod_is_ofdm": ["B2"],
    "df_mod_is_noise_none": ["B2", "B4"],
    "df_pulse_repetition_interval_ms": ["B2"],

    # B3: thermal and shape
    "tv_bbox_width": ["B3"],
    "tv_bbox_height": ["B3"],
    "tv_aspect_ratio": ["B3"],
    "tv_normalized_pixel_area": ["B3"],
    "tv_detector_confidence": ["B3", "B4"],
    "it_bbox_width": ["B3"],
    "it_bbox_height": ["B3"],
    "it_aspect_ratio": ["B3"],
    "it_target_mean_temp_c": ["B3"],
    "it_background_temp_c": ["B3", "B4"],
    "it_temp_contrast_delta_t_c": ["B3"],
    "it_peak_intensity_norm": ["B3"],
    "it_thermal_spatial_variance": ["B3"],
    "it_hotspot_count": ["B3"],
    "it_detector_confidence": ["B3", "B4"],

    # B4: environment and measurement context/reliability
    "radar_detection_confidence": ["B4"],
    "pan_background_clutter_index": ["B4"],
    "pan_ambient_light_level_norm": ["B4"],
    "pan_visibility_haze_index_norm": ["B4"],
    "pan_track_duration_sec": ["B4"],
}

BOTTLENECK_NAMES = ["B1", "B2", "B3", "B4"]
SENSOR_PREFIXES = ("radar_", "df_", "tv_", "it_", "pan_")


def build_routing_mask() -> torch.Tensor:
    """Return [4, 54] boolean mask; True means that connection is allowed."""
    missing = sorted(set(FEATURE_NAMES) - set(ROUTING_BY_NAME))
    extra = sorted(set(ROUTING_BY_NAME) - set(FEATURE_NAMES))
    if missing or extra:
        raise ValueError(f"Routing mismatch. Missing={missing}, extra={extra}")

    mask = torch.zeros(len(BOTTLENECK_NAMES), len(FEATURE_NAMES), dtype=torch.bool)
    bottleneck_to_row = {name: i for i, name in enumerate(BOTTLENECK_NAMES)}
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        destinations = ROUTING_BY_NAME[feature_name]
        if not destinations:
            raise ValueError(f"{feature_name} has no bottleneck destination")
        for destination in destinations:
            mask[bottleneck_to_row[destination], feature_idx] = True

    if not mask.any(dim=1).all():
        raise ValueError("Every bottleneck must receive at least one feature")
    return mask


def sensor_id_for_feature(name: str) -> int:
    for idx, prefix in enumerate(SENSOR_PREFIXES):
        if name.startswith(prefix):
            return idx
    raise ValueError(f"Unknown sensor prefix for feature: {name}")


class SpecializedBottleneckLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bottlenecks, feature_tokens, routing_mask):
        # routing_mask: [4, 54], True means the connection is permitted.
        # MultiheadAttention uses True in attn_mask to mean "blocked",
        # so invert the routing mask.
        blocked_mask = ~routing_mask

        cross, _ = self.cross_attention(
            query=bottlenecks,
            key=feature_tokens,
            value=feature_tokens,
            attn_mask=blocked_mask,
            need_weights=False,
        )
        bottlenecks = self.cross_norm(bottlenecks + self.dropout(cross))

        shared, _ = self.self_attention(
            query=bottlenecks,
            key=bottlenecks,
            value=bottlenecks,
            need_weights=False,
        )
        bottlenecks = self.self_norm(bottlenecks + self.dropout(shared))
        bottlenecks = self.ffn_norm(bottlenecks + self.ffn(bottlenecks))
        return bottlenecks


class SpecializedBottleneckTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.num_features = len(FEATURE_NAMES)
        self.d_model = d_model

        # Every scalar feature becomes its own token.
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Embedding(self.num_features, d_model)
        self.sensor_embedding = nn.Embedding(len(SENSOR_PREFIXES), d_model)

        sensor_ids = torch.tensor(
            [sensor_id_for_feature(name) for name in FEATURE_NAMES],
            dtype=torch.long,
        )
        self.register_buffer("sensor_ids", sensor_ids, persistent=True)

        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, len(BOTTLENECK_NAMES), d_model) * 0.02
        )
        self.register_buffer("routing_mask", build_routing_mask(), persistent=True)

        self.layers = nn.ModuleList([
            SpecializedBottleneckLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input shape [batch, {self.num_features}], got {tuple(x.shape)}"
            )

        batch_size = x.shape[0]
        feature_ids = torch.arange(self.num_features, device=x.device)

        feature_tokens = self.value_projection(x.unsqueeze(-1))
        feature_tokens = (
            feature_tokens
            + self.feature_embedding(feature_ids).unsqueeze(0)
            + self.sensor_embedding(self.sensor_ids).unsqueeze(0)
        )

        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            bottlenecks = layer(
                bottlenecks, feature_tokens, self.routing_mask
            )

        fused = bottlenecks.mean(dim=1)
        return self.classifier(fused)
