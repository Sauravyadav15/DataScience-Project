import pandas as pd
import numpy as np
import h5py
import os
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

HDF5_PATH      = "dataset.h5"
PREPROCESSED_DIR = r"D:\Vs_code\ELEC292_Project\data\preprocessed"

COL_MAP = {
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Linear Acceleration x (m/s^2)": "x",
    "Linear Acceleration y (m/s^2)": "y",
    "Linear Acceleration z (m/s^2)": "z",
}

CSV_FILES = {
    "Vishwas": {
        "Jumping": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Jump Back Pocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Jump Front Pocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Jump Leftpocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Jump Right Pocket.csv",
        ],
        "Walking": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Walk Back pocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Walk Front pocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Walk Left pocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Walk Right Pocket.csv",
        ],
    },
    "Saurav": {
        "Jumping": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_jumping_inhand.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_jumping_inhand(left).csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_jumping_leftpocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_jumping_rightpocketpant.csv",
        ],
        "Walking": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_walking_backpocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_walking_front(landscape HOld).csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_walking_leftpocketjacket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Saurav\Saurav_walking_upperjacket.csv",
        ],
    },
    "Tej": {
        "Jumping": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Jumping_BackPocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Jumping_FrontPocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Jumping_InHand.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Jumping_Jacket.csv",
        ],
        "Walking": [
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Walking_BackPocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Walking_FrontPocket.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Walking_InHand.csv",
            r"D:\Vs_code\ELEC292_Project\data\raw\Tej\Tej_Walking_Jacket.csv",
        ],
    },
}

# ── Step 1: Load & Inspect ────────────────────────────────────────────────────

def load_and_inspect(filepath):
    df = pd.read_csv(filepath).rename(columns=COL_MAP)[["x", "y", "z"]].astype(float)

    # print(f"  Shape   : {df.shape}")
    # print(f"  Missing : {df.isnull().sum().to_dict()}")

    return df

# ── Step 2: Handle Missing Values ─────────────────────────────────────────────

def handle_missing(df):
    df = df.interpolate(method="linear", limit=5)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)

    # if dropped > 0:
    #     print(f"  ⚠️  Dropped {dropped} rows with unfillable NaNs")
    # else:
    #     print(f"  ✅ No missing values")

    return df

# ── Step 3: Remove Outliers ───────────────────────────────────────────────────

def remove_outliers(df, z_threshold=3.0):
    for col in ["x", "y", "z"]:
        mean, std     = df[col].mean(), df[col].std()
        lower, upper  = mean - z_threshold * std, mean + z_threshold * std
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col]       = df[col].clip(lower, upper)

    #     if outlier_count > 0:
    #         print(f"  📌 {col}: clipped {outlier_count} outliers")

    # print(f"  ✅ Outlier clipping done")
    return df

# ── Step 4: Normalize ─────────────────────────────────────────────────────────

def normalize(df):
    df[["x", "y", "z"]] = StandardScaler().fit_transform(df[["x", "y", "z"]])
    # print(f"  ✅ Z-score normalization applied")
    return df

# ── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess(filepath):
    df = load_and_inspect(filepath)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = normalize(df)
    print(f"  ✅ Final shape: {df.shape}")
    return df

# ── Build HDF5 ────────────────────────────────────────────────────────────────

def build_hdf5():
    with h5py.File(HDF5_PATH, "w") as f:
        raw_group  = f.create_group("Raw data")
        prep_group = f.create_group("Pre-processed data")

        for member, actions in CSV_FILES.items():
            for action, files in actions.items():
                for filepath in files:

                    key = f"{member}/{action}/{filepath.split('/')[-1].replace('.csv', '')}"
                    print(f"\n📂 {key}")

                    raw = pd.read_csv(filepath).rename(columns=COL_MAP)[["x", "y", "z"]].values.astype(np.float32)
                    raw_group.create_dataset(key, data=raw, compression="gzip")

                    processed = preprocess(filepath).values.astype(np.float32)
                    prep_group.create_dataset(key, data=processed, compression="gzip")

                    print(f"  💾 raw{raw.shape}  →  prep{processed.shape}")

    print(f"\n🎉 HDF5 file ready: {HDF5_PATH}")

# ── Save Preprocessed CSVs to Disk ── (NEW)──────────────────────────────────

def save_preprocessed_csvs():
    for member, actions in CSV_FILES.items():
        for action, files in actions.items():
            for filepath in files:

                filename = os.path.basename(filepath)
                save_dir = os.path.join(PREPROCESSED_DIR, member, action.capitalize())

                os.makedirs(save_dir, exist_ok=True)

                df = preprocess(filepath)
                df.to_csv(os.path.join(save_dir, filename), index=False)

                print(f"  💾 Saved → {os.path.join(save_dir, filename)}")

    print(f"\n🎉 All preprocessed CSVs saved to: {PREPROCESSED_DIR}")

# ── Verify HDF5 Structure ─────────────────────────────────────────────────────

def verify_hdf5():
    print(f"\n📦 HDF5 structure:\n")
    with h5py.File(HDF5_PATH, "r") as f:
        def print_tree(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name}   shape={obj.shape}")
            else:
                print(f"{indent}📁 {name}/")
        f.visititems(print_tree)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_hdf5()
    save_preprocessed_csvs()
    verify_hdf5()

    df = pd.read_csv(
        r"D:\Vs_code\ELEC292_Project\data\raw\Vishwas\Walk Back pocket.csv"
    )
    time_diffs = df["Time (s)"].diff().dropna()
    avg_interval = time_diffs.mean()
    sampling_rate = 1 / avg_interval
    print(f"Avg time interval : {avg_interval:.6f} seconds")
    print(f"Sampling rate     : {sampling_rate:.2f} Hz")
    print(f"5-second window   : {int(sampling_rate * 5)} samples")

