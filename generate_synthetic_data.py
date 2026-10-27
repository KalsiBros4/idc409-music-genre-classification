"""
generate_synthetic_data.py
--------------------------
Generates a synthetic features_30_sec.csv that mirrors the GTZAN schema
exactly (same column names, 1000 rows, 10 genres).

This is used ONLY to execute the notebook end-to-end and verify the pipeline.
The synthetic data is statistically plausible but not the real GTZAN dataset.

The real dataset must be downloaded from Kaggle for genuine results.
Run:
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
N_PER_GENRE = 100

# Genre-specific means for key features (to create realistic separation)
GENRE_PARAMS = {
    #              mfcc1,  tempo,  centroid, zero_cr,  rms,   chroma
    "blues":      (-185,   112,    1750,     0.083,   0.115, 0.34),
    "classical":  (-250,    88,    1200,     0.052,   0.075, 0.40),
    "country":    (-195,   120,    1900,     0.090,   0.120, 0.36),
    "disco":      (-175,   120,    2100,     0.100,   0.140, 0.37),
    "hiphop":     (-160,    95,    1800,     0.085,   0.130, 0.33),
    "jazz":       (-210,   108,    1500,     0.068,   0.095, 0.38),
    "metal":      (-145,   142,    2500,     0.145,   0.175, 0.30),
    "pop":        (-175,   118,    2050,     0.095,   0.135, 0.36),
    "reggae":     (-168,    90,    1720,     0.082,   0.125, 0.35),
    "rock":       (-160,   132,    2200,     0.130,   0.155, 0.32),
}

rows = []
for genre in GENRES:
    m1, tempo_m, cent_m, zcr_m, rms_m, chroma_m = GENRE_PARAMS[genre]
    for i in range(N_PER_GENRE):
        filename = f"{genre}.{i:05d}.wav"
        row = {"filename": filename, "length": 661794}

        row["chroma_stft_mean"] = float(np.clip(rng.normal(chroma_m, 0.05), 0.1, 0.8))
        row["chroma_stft_var"]  = float(rng.uniform(0.05, 0.12))
        row["rms_mean"]         = float(np.clip(rng.normal(rms_m, 0.02), 0.01, 0.4))
        row["rms_var"]          = float(rng.uniform(0.001, 0.01))
        row["spectral_centroid_mean"] = float(np.clip(rng.normal(cent_m, 300), 500, 5000))
        row["spectral_centroid_var"]  = float(rng.uniform(100000, 500000))
        row["spectral_bandwidth_mean"] = float(np.clip(rng.normal(2000, 300), 800, 4000))
        row["spectral_bandwidth_var"]  = float(rng.uniform(50000, 300000))
        row["rolloff_mean"]    = float(np.clip(rng.normal(cent_m * 1.8, 400), 1000, 8000))
        row["rolloff_var"]     = float(rng.uniform(500000, 2000000))
        row["zero_crossing_rate_mean"] = float(np.clip(rng.normal(zcr_m, 0.015), 0.01, 0.3))
        row["zero_crossing_rate_var"]  = float(rng.uniform(0.0005, 0.003))
        row["harmony_mean"]    = float(rng.normal(0.0, 0.01))
        row["harmony_var"]     = float(rng.uniform(0.0001, 0.001))
        row["perceptr_mean"]   = float(rng.normal(0.0, 0.01))
        row["perceptr_var"]    = float(rng.uniform(0.0001, 0.001))
        row["tempo"]           = float(np.clip(rng.normal(tempo_m, 15), 50, 220))

        for k in range(1, 21):
            scale = max(1, 60 - k * 3)
            row[f"mfcc{k}_mean"] = float(rng.normal(m1 / (k * 0.7 + 0.3), scale))
            row[f"mfcc{k}_var"]  = float(rng.uniform(50, 800))

        row["label"] = genre
        rows.append(row)

df = pd.DataFrame(rows)

# Verify column order matches expected GTZAN layout
assert "filename" in df.columns
assert "label" in df.columns
assert df.shape == (1000, 60), f"Unexpected shape: {df.shape}"

out_path = os.path.join("data", "features_30_sec.csv")
os.makedirs("data", exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Synthetic dataset written: {out_path}")
print(f"Shape: {df.shape}")
print(f"Genres: {df['label'].value_counts().to_dict()}")
print()
print("NOTE: This is SYNTHETIC data for pipeline verification only.")
print("Replace with the real GTZAN CSV for genuine results.")
