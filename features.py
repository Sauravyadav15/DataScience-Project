import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

# ── Config ────────────────────────────────────────────────────────────────────

HDF5_PATH     = r"D:\Vs_code\ELEC292_Project\dataset.h5"
SAMPLING_RATE = 400
MODEL_DIR     = Path(__file__).resolve().parent / "models"

# ── Extract 18 features from one axis ────────────────────────────────────────

def extract_features(signal):
    # Time domain
    fts = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.max(signal) - np.min(signal),
        np.median(signal),
        np.var(signal),
        skew(signal),
        kurtosis(signal),
        np.sqrt(np.mean(signal ** 2)),                      # RMS
        np.sum(np.diff(np.sign(np.diff(signal))) < 0),      # peaks
        np.sum(signal ** 2),                                 # energy
        np.sum(np.diff(np.sign(signal)) != 0),               # zero crossing rate
        np.percentile(signal, 25),
        np.percentile(signal, 75),
    ]

    # Frequency domain (FFT)
    fft_vals         = np.abs(np.fft.fft(signal))[:len(signal)//2]
    freqs            = np.fft.fftfreq(len(signal), d=1/SAMPLING_RATE)[:len(signal)//2]
    dominant_freq    = freqs[np.argmax(fft_vals)]
    spectral_energy  = np.sum(fft_vals ** 2)
    psd              = fft_vals ** 2 / (np.sum(fft_vals ** 2) + 1e-10)
    spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))

    fts.extend([dominant_freq, spectral_energy, spectral_entropy])

    return fts    # 18 features per axis

# ── Extract features from all 3 axes of one window ───────────────────────────

def extract_window_features(window):
    features = []
    for axis in range(3):
        features.extend(extract_features(window[:, axis]))
    return features    # 54 features total (18 × 3 axes)

# ── Process all windows ───────────────────────────────────────────────────────

def build_feature_matrix(X):
    return np.array([extract_window_features(window) for window in X])

# ── Load, extract, normalize, save ───────────────────────────────────────────

def run():
    with h5py.File(HDF5_PATH, "r") as f:
        X_train = f["Segmented data/Train/X_train"][:]
        X_test  = f["Segmented data/Test/X_test"][:]
        y_train = f["Segmented data/Train/y_train"][:]
        y_test  = f["Segmented data/Test/y_test"][:]

    print("Extracting features...")
    F_train = build_feature_matrix(X_train)
    F_test  = build_feature_matrix(X_test)
    print(f"  Train features : {F_train.shape}  →  (windows, features)")
    print(f"  Test features  : {F_test.shape}")

    # Normalize — fit on train only, apply to both
    scaler  = MinMaxScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)
    print("  ✅ Min-max normalization applied")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    print(f"  💾 Scaler saved to {MODEL_DIR / 'scaler.joblib'}")


    # Save back to HDF5
    with h5py.File(HDF5_PATH, "a") as f:
        for key in ["Features/F_train", "Features/F_test",
                    "Features/y_train", "Features/y_test"]:
            if key in f:
                del f[key]

        f.create_dataset("Features/F_train", data=F_train.astype(np.float32), compression="gzip")
        f.create_dataset("Features/F_test",  data=F_test.astype(np.float32),  compression="gzip")
        f.create_dataset("Features/y_train", data=y_train, compression="gzip")
        f.create_dataset("Features/y_test",  data=y_test,  compression="gzip")

    print(f"\n  💾 Saved to HDF5")
    print(f"     F_train : {F_train.shape}")
    print(f"     F_test  : {F_test.shape}")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
