import pandas as pd
import numpy as np
import h5py
import os
from sklearn.utils import shuffle

# ── Config ─────────────────────────────────────────────────────────────

HDF5_PATH    = r"D:\Vs_code\ELEC292_Project\dataset.h5"
PREPROCESSED = r"D:\Vs_code\ELEC292_Project\data\preprocessed"
WINDOW_SIZE  = 500    # 5 sec × 400 Hz
STEP_SIZE    = 250    # slide by 1000 rows (50% overlap)
LABEL_MAP    = {"walking": 0, "jumping": 1}

# ── Collect all windows from every preprocessed CSV ──────────────────────────

def collect_windows():
    all_windows, all_labels = [], []

    for member in os.listdir(PREPROCESSED):
        for action in os.listdir(os.path.join(PREPROCESSED, member)):
            label  = LABEL_MAP[action.lower()]
            folder = os.path.join(PREPROCESSED, member, action)

            for filename in os.listdir(folder):
                signal    = pd.read_csv(os.path.join(folder, filename))[["x","y","z"]].values
                n_windows = (len(signal) - WINDOW_SIZE) // STEP_SIZE + 1

                for i in range(n_windows):
                    start = i * STEP_SIZE
                    all_windows.append(signal[start : start + WINDOW_SIZE])
                    all_labels.append(label)

                print(f"  ✅ {member}/{action}/{filename}  →  {n_windows} windows")

    return np.array(all_windows), np.array(all_labels)

# ── Shuffle and split 90/10 ───────────────────────────────────────────────────

def split(windows, labels):
    windows, labels = shuffle(windows, labels, random_state=42)
    cut             = int(len(windows) * 0.9)

    print(f"\n  Total   : {len(windows)} windows")
    print(f"  Train   : {cut}  |  Test : {len(windows) - cut}")

    return windows[:cut], windows[cut:], labels[:cut], labels[cut:]

# ── Save to HDF5 ──────────────────────────────────────────────────────────────

def save(X_train, X_test, y_train, y_test):
    with h5py.File(HDF5_PATH, "a") as f:
        if "Segmented data" in f:
            del f["Segmented data"]

        seg = f.create_group("Segmented data")
        seg.create_dataset("Train/X_train", data=X_train.astype(np.float32), compression="gzip")
        seg.create_dataset("Train/y_train", data=y_train.astype(np.int8),    compression="gzip")
        seg.create_dataset("Test/X_test",   data=X_test.astype(np.float32),  compression="gzip")
        seg.create_dataset("Test/y_test",   data=y_test.astype(np.int8),     compression="gzip")

    print(f"\n  💾 X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"     X_test  {X_test.shape}   y_test  {y_test.shape}")
    print(f"\n  🎉 Saved to {HDF5_PATH}")

# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    print("📂 Collecting windows...\n")
    windows, labels = collect_windows()
    X_train, X_test, y_train, y_test = split(windows, labels)
    save(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()