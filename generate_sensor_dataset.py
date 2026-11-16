import csv
import math
import random
import struct
import zipfile


rng = random.Random(42)


def choose(values, p=None):
    if p is None:
        return rng.choice(values)
    return rng.choices(values, weights=p, k=1)[0]


def clip(value, lower, upper):
    return min(max(value, lower), upper)


class RandomHelpers:
    choice = staticmethod(choose)
    uniform = staticmethod(rng.uniform)
    normal = staticmethod(rng.gauss)


class MathHelpers:
    random = RandomHelpers()
    log10 = staticmethod(math.log10)
    clip = staticmethod(clip)


# Keep the simulation expressions below concise without third-party packages.
np = MathHelpers()


def write_npy(zip_file, name, rows, dtype):
    if rows and isinstance(rows[0], list):
        shape = (len(rows), len(rows[0]))
        values = [value for row in rows for value in row]
    else:
        shape = (len(rows),)
        values = rows

    descr, pack_code = ('<f4', 'f') if dtype == 'float32' else ('<i8', 'q')
    shape_text = ', '.join(str(size) for size in shape)
    if len(shape) == 1:
        shape_text += ','
    header = (
        f"{{'descr': '{descr}', 'fortran_order': False, "
        f"'shape': ({shape_text}), }}"
    ).encode('ascii')
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header += b' ' * padding + b'\n'
    payload = (
        b'\x93NUMPY'
        + b'\x01\x00'
        + struct.pack('<H', len(header))
        + header
        + struct.pack(f'<{len(values)}{pack_code}', *values)
    )
    zip_file.writestr(f'{name}.npy', payload)


def save_npz(path, arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, rows, dtype in arrays:
            write_npy(archive, name, rows, dtype)

TOTAL_SAMPLES = 5000
CLASSES = ['Drone', 'Bird', 'Clutter']
samples_distribution = {'Drone': 1700, 'Bird': 1700, 'Clutter': 1600}

# ==============================================================================
# 54 STANDARDIZED DESCRIPTIVE COLUMN NAMES
# ==============================================================================
radar_cols = [
    'radar_slant_range_m', 'radar_azimuth_deg', 'radar_elevation_deg',
    'radar_radial_velocity_mps', 'radar_radial_acceleration_mps2',
    'radar_trajectory_curvature', 'radar_rcs_mean_dbsm', 'radar_rcs_variance',
    'radar_micro_doppler_peak_hz', 'radar_doppler_bandwidth_hz',
    'radar_snr_db', 'radar_detection_confidence'
]

df_cols = [
    'df_aoa_azimuth_deg', 'df_aoa_elevation_deg', 'df_bearing_rate_degps',
    'df_carrier_frequency_mhz', 'df_bandwidth_mhz', 'df_hopping_rate_hops_sec',
    'df_rssi_dbm', 'df_snr_db', 'df_mod_is_fhss', 'df_mod_is_dsss',
    'df_mod_is_ofdm', 'df_mod_is_noise_none', 'df_pulse_repetition_interval_ms'
]

tv_cols = [
    'tv_bbox_center_x', 'tv_bbox_center_y', 'tv_bbox_width', 'tv_bbox_height',
    'tv_aspect_ratio', 'tv_normalized_pixel_area', 'tv_optical_flow_vx',
    'tv_optical_flow_vy', 'tv_detector_confidence'
]

it_cols = [
    'it_bbox_center_x', 'it_bbox_center_y', 'it_bbox_width', 'it_bbox_height',
    'it_aspect_ratio', 'it_target_mean_temp_c', 'it_background_temp_c',
    'it_temp_contrast_delta_t_c', 'it_peak_intensity_norm',
    'it_thermal_spatial_variance', 'it_hotspot_count', 'it_detector_confidence'
]

pan_cols = [
    'pan_mount_azimuth_deg', 'pan_mount_elevation_deg', 'pan_mount_slew_rate_az_degps',
    'pan_mount_slew_rate_el_degps', 'pan_background_clutter_index',
    'pan_ambient_light_level_norm', 'pan_visibility_haze_index_norm',
    'pan_track_duration_sec'
]

all_columns = radar_cols + df_cols + tv_cols + it_cols + pan_cols + ['target_class_id', 'target_class_name']

data_rows = []

for class_id, class_name in enumerate(CLASSES):
    num_class_samples = samples_distribution[class_name]
    
    for _ in range(num_class_samples):
        # -------------------------------------------------------------
        # GLOBAL ENVIRONMENT & TARGET PHYSICAL DIMENSIONS
        # -------------------------------------------------------------
        # Day (light ~ 0.5-1.0) vs Night (Full Moon, light ~ 0.05-0.2)
        is_night = np.random.choice([False, True], p=[0.7, 0.3])
        light_lux = np.random.uniform(0.05, 0.20) if is_night else np.random.uniform(0.50, 1.0)
        ambient_temp = np.random.uniform(14, 22) if is_night else np.random.uniform(24, 34)
        visibility = np.random.uniform(0.6, 1.0)

        # Target physical size (meters)
        if class_name == 'Drone':
            # 80% small drone (0.3m x 0.3m Mavic), 20% tactical (1.6m x 1.6m MEAT)
            is_large = np.random.choice([False, True], p=[0.8, 0.2])
            target_size_m = 1.6 if is_large else 0.3
            # Radar constraints: min 200m; up to 5km (small) or 7.5km (tactical)
            max_r = 7500.0 if is_large else 5000.0
            dist_range = np.random.uniform(200.0, max_r)
        elif class_name == 'Bird':
            is_large = False
            target_size_m = np.random.uniform(0.2, 0.4) # ~0.3m average
            dist_range = np.random.uniform(200.0, 3500.0) # Birds tracked within 3.5 km
        else: # Clutter
            is_large = False
            target_size_m = np.random.uniform(0.3, 2.0)
            dist_range = np.random.uniform(200.0, 5000.0)

        azimuth = np.random.uniform(-180, 180)
        elevation = np.random.uniform(5, 45)

        # -------------------------------------------------------------
        # 1. LLLWR (RADAR) - 12 PARAMETERS (Min Range: 200m)
        # -------------------------------------------------------------
        if class_name == 'Drone':
            r_vel = np.random.normal(14, 4)
            r_acc = np.random.normal(2.0, 0.8)
            r_curv = np.random.uniform(0.1, 0.6)
            # RCS: 0.001 - 0.01 m^2 (-30 to -20 dBsm) for 0.3m; 0.1 - 0.5 m^2 (-10 to -3 dBsm) for 1.6m
            rcs_mean = np.random.uniform(-9, -4) if is_large else np.random.uniform(-28, -21)
            rcs_var = np.random.uniform(0.5, 2.0)
            doppler_peak = np.random.normal(160, 20) # High-speed rotor blade micro-Doppler
            doppler_bw = np.random.normal(35, 5)
            # Radar SNR equation decay: SNR ~ RCS / Range^4
            base_snr = 35 if is_large else 28
            r_snr = base_snr - 40 * np.log10(dist_range / 1000.0) + np.random.normal(0, 1.5)
            r_snr = max(float(r_snr), 3.0)
            r_conf = np.clip(0.99 - (dist_range / max_r) * 0.15, 0.70, 0.99)

        elif class_name == 'Bird':
            r_vel = np.random.normal(8, 2.5)
            r_acc = np.random.normal(0.5, 0.2)
            r_curv = np.random.uniform(0.05, 0.25)
            # Bird RCS: 0.002 - 0.015 m^2 (-27 to -18 dBsm)
            rcs_mean = np.random.uniform(-27, -19)
            rcs_var = np.random.uniform(1.2, 3.0)
            doppler_peak = np.random.normal(14, 3) # Wing flap: 10-18 Hz
            doppler_bw = np.random.normal(8, 2)
            r_snr = 25 - 40 * np.log10(dist_range / 1000.0) + np.random.normal(0, 1.5)
            r_snr = max(float(r_snr), 2.0)
            r_conf = np.clip(0.85 - (dist_range / 3500.0) * 0.25, 0.50, 0.88)

        else: # Clutter
            r_vel = np.random.normal(0, 0.8)
            r_acc = 0.0
            r_curv = 0.0
            rcs_mean = np.random.uniform(-35, -20)
            rcs_var = np.random.uniform(3.0, 8.0)
            doppler_peak = np.random.normal(1.2, 0.5)
            doppler_bw = np.random.normal(1.5, 0.5)
            r_snr = np.random.uniform(3, 8)
            r_conf = np.random.uniform(0.10, 0.35)

        radar_features = [
            dist_range, azimuth, elevation, abs(r_vel), abs(r_acc), r_curv,
            rcs_mean, rcs_var, abs(doppler_peak), doppler_bw, r_snr, r_conf
        ]

        # -------------------------------------------------------------
        # 2. DF (RF DIRECTION FINDER) - 13 PARAMETERS
        # -------------------------------------------------------------
        # Mavic class (20-26 dBm, <= 8km) | Military class (27-33 dBm, <= 15km)
        if class_name == 'Drone':
            tx_power_dbm = np.random.uniform(28, 33) if is_large else np.random.uniform(20, 26)
            df_max_range = 15000.0 if is_large else 8000.0
            
            if dist_range <= df_max_range:
                df_aoa_az = azimuth + np.random.normal(0, 1.2)
                df_aoa_el = elevation + np.random.normal(0, 1.8)
                df_bearing_rate = np.random.uniform(0.2, 1.2)
                df_freq = np.random.choice([2412, 2462, 5745, 5825, 915, 433])
                df_bw = np.random.choice([10.0, 20.0, 40.0])
                df_hop_rate = np.random.uniform(400, 1200) # FHSS
                # Free-space path loss formula for RSSI
                path_loss = 20 * np.log10(dist_range) + 20 * np.log10(df_freq) - 27.55
                df_rssi = float(np.clip(tx_power_dbm - path_loss, -100.0, -45.0))
                df_snr = float(np.clip(df_rssi + 105.0, 3.0, 30.0))
                df_mod = [1, 0, 0, 0] # FHSS active
                df_pri = np.random.uniform(2.0, 8.0)
            else:
                # Out of DF detection range
                df_aoa_az, df_aoa_el, df_bearing_rate = 0.0, 0.0, 0.0
                df_freq, df_bw, df_hop_rate = 0.0, 0.0, 0.0
                df_rssi, df_snr = -105.0, 1.0
                df_mod = [0, 0, 0, 1]
                df_pri = 0.0
        else: # Bird or Clutter (No RF emissions)
            df_aoa_az = np.random.uniform(-180, 180)
            df_aoa_el = np.random.uniform(0, 45)
            df_bearing_rate = 0.0
            df_freq, df_bw, df_hop_rate = 0.0, 0.0, 0.0
            df_rssi = -105.0 # Noise floor
            df_snr = np.random.uniform(0.5, 2.0)
            df_mod = [0, 0, 0, 1] # None / Noise
            df_pri = 0.0

        df_features = [
            df_aoa_az, df_aoa_el, df_bearing_rate, df_freq, df_bw,
            df_hop_rate, df_rssi, df_snr, df_mod[0], df_mod[1], df_mod[2], df_mod[3], df_pri
        ]

        # -------------------------------------------------------------
        # 3. TV (DAYLIGHT OPTICAL CAMERA) - 9 PARAMETERS
        # -------------------------------------------------------------
        # 0.3m target <= 3.5km; 1.6m target <= 6.0km
        tv_ti_max_range = 6000.0 if (target_size_m >= 1.0) else 3500.0
        tv_visible = (dist_range <= tv_ti_max_range) and (not is_night) and (visibility > 0.4)

        tv_x = np.clip(np.random.normal(0.5, 0.08), 0.05, 0.95)
        tv_y = np.clip(np.random.normal(0.5, 0.08), 0.05, 0.95)

        # Angular pixel projection scaling with distance: size / distance
        norm_scale = float(np.clip((target_size_m / dist_range) * 20.0, 0.005, 0.12))

        if tv_visible:
            if class_name == 'Drone':
                tv_w = norm_scale
                tv_h = norm_scale * (0.6 if is_large else 0.85)
                tv_asp = tv_w / tv_h
                tv_area = tv_w * tv_h
                tv_vx, tv_vy = np.random.normal(0.015, 0.003), np.random.normal(0.005, 0.002)
                tv_conf = float(np.clip(0.96 - (dist_range / tv_ti_max_range) * 0.30, 0.65, 0.98))
            elif class_name == 'Bird':
                tv_w = norm_scale * 1.1
                tv_h = norm_scale
                tv_asp = tv_w / tv_h
                tv_area = tv_w * tv_h
                tv_vx, tv_vy = np.random.normal(0.010, 0.003), np.random.normal(0.007, 0.002)
                tv_conf = float(np.clip(0.85 - (dist_range / tv_ti_max_range) * 0.35, 0.45, 0.88))
            else: # Clutter
                tv_w, tv_h = 0.03, 0.03
                tv_asp, tv_area = 1.0, 0.0009
                tv_vx, tv_vy = 0.0, 0.0
                tv_conf = np.random.uniform(0.05, 0.25)
        else:
            # Blinded by night or beyond 3.5km/6.0km limit
            tv_w, tv_h, tv_asp, tv_area = 0.001, 0.001, 1.0, 0.000001
            tv_vx, tv_vy = 0.0, 0.0
            tv_conf = 0.02

        tv_features = [tv_x, tv_y, tv_w, tv_h, tv_asp, tv_area, tv_vx, tv_vy, tv_conf]

        # -------------------------------------------------------------
        # 4. IT (THERMAL IMAGER / TI) - 12 PARAMETERS
        # -------------------------------------------------------------
        # TI works in Day & Night, but limited by same optical resolution: 0.3m <= 3.5km, 1.6m <= 6km
        ti_detectable = (dist_range <= tv_ti_max_range)
        it_x, it_y = tv_x + np.random.normal(0, 0.001), tv_y + np.random.normal(0, 0.001)

        if ti_detectable:
            it_w, it_h = tv_w, tv_h
            it_asp = it_w / max(it_h, 1e-4)
            if class_name == 'Drone':
                delta_t = np.random.normal(18, 3) if is_large else np.random.normal(12, 2)
                it_mean = ambient_temp + delta_t
                it_peak = 0.88
                it_var = np.random.uniform(4.5, 8.0)
                it_hotspots = 2.0 if is_large else 4.0 # engine vs 4 motors
                it_conf = float(np.clip(0.97 - (dist_range / tv_ti_max_range) * 0.25, 0.65, 0.98))
            elif class_name == 'Bird':
                delta_t = np.random.normal(5.0, 1.0) # Biological heat
                it_mean = ambient_temp + delta_t
                it_peak = 0.40
                it_var = np.random.uniform(0.6, 1.4)
                it_hotspots = 0.0
                it_conf = float(np.clip(0.85 - (dist_range / tv_ti_max_range) * 0.30, 0.45, 0.86))
            else: # Clutter
                delta_t = np.random.normal(0.4, 0.3)
                it_mean = ambient_temp + delta_t
                it_peak = 0.12
                it_var = np.random.uniform(0.05, 0.25)
                it_hotspots = 0.0
                it_conf = np.random.uniform(0.05, 0.25)
        else:
            # Beyond 3.5km/6.0km thermal tracking range
            it_w, it_h, it_asp = 0.001, 0.001, 1.0
            it_mean = ambient_temp
            delta_t = 0.0
            it_peak = 0.05
            it_var = 0.01
            it_hotspots = 0.0
            it_conf = 0.01

        it_features = [
            it_x, it_y, it_w, it_h, it_asp, it_mean, ambient_temp,
            delta_t, it_peak, it_var, it_hotspots, it_conf
        ]

        # -------------------------------------------------------------
        # 5. PAN (PANORAMIC CAMERA) - 8 PARAMETERS
        # -------------------------------------------------------------
        # Spec limits:
        # Day: 0.3m <= 1.5km; 1.6m <= 5.0km
        # Night: 0.3m <= 1.0km; 1.6m <= 3.0km
        if is_night:
            pan_limit = 3000.0 if (target_size_m >= 1.0) else 1000.0
        else:
            pan_limit = 5000.0 if (target_size_m >= 1.0) else 1500.0

        pan_detected = (dist_range <= pan_limit) and (class_name != 'Clutter')

        pan_mount = azimuth + np.random.normal(0, 0.5)
        tilt_mount = elevation + np.random.normal(0, 0.5)
        slew_pan = np.random.uniform(0.2, 1.8)
        slew_tilt = np.random.uniform(0.1, 0.8)
        clutter_idx = 0.0 if (class_name != 'Clutter') else np.random.choice([1.0, 2.0])
        track_dur = np.random.uniform(5.0, 40.0) if pan_detected else np.random.uniform(0.5, 2.5)

        pan_features = [
            pan_mount, tilt_mount, slew_pan, slew_tilt,
            clutter_idx, light_lux, visibility, track_dur
        ]

        # -------------------------------------------------------------
        # ROW ASSEMBLY
        # -------------------------------------------------------------
        full_sample = radar_features + df_features + tv_features + it_features + pan_features + [class_id, class_name]
        data_rows.append(full_sample)

# Shuffle deterministically, then write the CSV and NumPy-compatible archive.
random.Random(42).shuffle(data_rows)

with open('sensor_data_calibrated_3classes_5000.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(all_columns)
    writer.writerows(data_rows)
print("Saved 5000 samples to 'sensor_data_calibrated_3classes_5000.csv'!")

save_npz(
    'sensor_data_calibrated_3classes_5000.npz',
    [
        ('radar', [row[0:12] for row in data_rows], 'float32'),
        ('df', [row[12:25] for row in data_rows], 'float32'),
        ('tv', [row[25:34] for row in data_rows], 'float32'),
        ('it', [row[34:46] for row in data_rows], 'float32'),
        ('pan', [row[46:54] for row in data_rows], 'float32'),
        ('labels', [row[54] for row in data_rows], 'int64'),
    ],
)
print("Saved compressed NumPy arrays to 'sensor_data_calibrated_3classes_5000.npz'!")