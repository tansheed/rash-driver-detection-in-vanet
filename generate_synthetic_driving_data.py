
"""
Generate synthetic SUMO-style vehicle trajectories for LSTM/GRU.

Creates:
  synthetic_driving_data.csv
  synthetic_lstm_windows.npz

Synthetic data is for pipeline testing only.
It is not actual SUMO output or real-world validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# Configuration
dt = 0.5
window_size = 40
n_trips = 500
trip_seconds = 120

raw_parts = []
window_parts = []
label_parts = []
trip_id_parts = []


def smooth_noise(n, scale, width=9):
    noise = rng.normal(0, scale, n)
    return np.convolve(
        noise, np.ones(width) / width, mode="same"
    )


n = int(trip_seconds / dt) + 1
times = np.arange(n) * dt

for trip_no in range(1, n_trips + 1):

    trip_id = f"TRIP_{trip_no:04d}"

    # 1. Generate speed in km/h
    speed = rng.uniform(35, 65) + smooth_noise(n, 5, 13)
    speed += 2 * np.sin(
        np.linspace(0, rng.uniform(1, 3) * np.pi, n)
    )

    labels = np.zeros(n, dtype=np.int64)

    # 2. Generate a smooth road-following heading.
    # SUMO convention: 0 = north, 90 = east,
    # 180 = south, 270 = west.
    heading_command = np.empty(n)
    heading_command[0] = rng.uniform(0, 360)

    heading_changes = smooth_noise(n, 1.0, 11)
    heading_command[1:] = (
        heading_command[0]
        + np.cumsum(heading_changes[1:])
    )

    # 3. Add anomalous maneuvers
    for _ in range(int(rng.integers(2, 6))):

        duration = int(rng.integers(8, 25))
        start = int(rng.integers(10, n - duration - 10))
        end = start + duration

        labels[start:end] = 1
        pattern = int(rng.integers(0, 3))
        t = np.arange(duration)

        # Abrupt, oscillating heading changes
        if pattern in (0, 2):
            direction = rng.choice([-1, 1])
            heading_command[start:end] += direction * (
                rng.uniform(10, 22)
                * np.sin(
                    2 * np.pi * t / max(4, duration // 2)
                )
            )

        # Acceleration and speed fluctuations
        if pattern in (1, 2):
            speed[start:end] += np.linspace(
                0, rng.uniform(15, 30), duration
            )
            speed[start:end] += rng.normal(0, 3, duration)

    speed = np.clip(speed, 5, 130)
    heading_command %= 360

    # 4. Integrate speed and heading to create positions.
    # These are local synthetic coordinates in metres.
    speed_mps = speed / 3.6
    theta = np.deg2rad(heading_command)

    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        x[i] = (
            x[i - 1]
            + speed_mps[i - 1] * dt * np.sin(theta[i - 1])
        )
        y[i] = (
            y[i - 1]
            + speed_mps[i - 1] * dt * np.cos(theta[i - 1])
        )

    # 5. Derive heading from consecutive positions.
    # atan2(dx, dy) matches SUMO's north-zero clockwise convention.
    dx = np.diff(x)
    dy = np.diff(y)

    heading = np.empty(n, dtype=np.float64)
    heading[0] = heading_command[0]
    heading[1:] = np.degrees(np.arctan2(dx, dy)) % 360

    # Encode the circular angle as two continuous features.
    heading_rad = np.deg2rad(heading)
    heading_sin = np.sin(heading_rad)
    heading_cos = np.cos(heading_rad)

    # 6. Save the raw time series.
    df = pd.DataFrame({
        "timestamp": times,
        "vehicle_id": trip_id,
        "x_position": np.round(x, 3),
        "y_position": np.round(y, 3),
        "vehicle_speed": np.round(speed, 2),
        "heading_angle": np.round(heading, 2),
        "label": labels
    })

    raw_parts.append(df)

    # 7. Build LSTM windows.
    # Features: speed, heading_sin, heading_cos.
    features = np.column_stack([
        speed,
        heading_sin,
        heading_cos
    ]).astype(np.float32)

    windows = np.stack([
        features[s:s + window_size]
        for s in range(n - window_size + 1)
    ])

    window_labels = np.array([
        int(labels[s:s + window_size].mean() >= 0.25)
        for s in range(n - window_size + 1)
    ], dtype=np.int64)

    window_parts.append(windows)
    label_parts.append(window_labels)

    trip_id_parts.append(
        np.full(len(window_labels), trip_id)
    )

    if trip_no % 50 == 0:
        print(f"Generated {trip_no}/{n_trips} trips")


# 8. Combine and save datasets.
raw = pd.concat(raw_parts, ignore_index=True)

X = np.concatenate(window_parts).astype(np.float32)
labels = np.concatenate(label_parts).astype(np.int64)
trip_ids = np.concatenate(trip_id_parts)

raw.to_csv("synthetic_driving_data.csv", index=False)

np.savez_compressed(
    "synthetic_lstm_windows.npz",
    X=X,
    y=labels,
    trip_ids=trip_ids
)

print("\nGeneration complete.")
print("Raw rows:", len(raw))
print("Independent trips:", raw["vehicle_id"].nunique())
print("X shape:", X.shape)
print("y shape:", labels.shape)
print("Trip IDs shape:", trip_ids.shape)
print("Window label counts:")
print(pd.Series(labels).value_counts().sort_index())
print("\nInput features: speed, heading_sin, heading_cos")
print("NPZ keys:", ["X", "y", "trip_ids"])
