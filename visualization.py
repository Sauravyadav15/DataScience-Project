import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

# ── Config ────────────────────────────────────────────────────────────────────

HDF5_PATH     = r"D:\Vs_code\ELEC292_Project\dataset.h5"
RAW_DIR       = r"D:\Vs_code\ELEC292_Project\data\raw"
PREP_DIR      = r"D:\Vs_code\ELEC292_Project\data\preprocessed"
SAMPLING_RATE = 400
TIME_500      = np.arange(500) / SAMPLING_RATE
COLORS        = ["#e74c3c", "#2ecc71", "#3498db"]
AXES          = ["X", "Y", "Z"]

COL_MAP = {
    "Acceleration x (m/s^2)": "x", "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Linear Acceleration x (m/s^2)": "x", "Linear Acceleration y (m/s^2)": "y",
    "Linear Acceleration z (m/s^2)": "z",
}

# ── Helper: load one window per class ─────────────────────────────────────────

def load_windows():
    with h5py.File(HDF5_PATH, "r") as f:
        X = f["Segmented data/Train/X_train"][:]
        y = f["Segmented data/Train/y_train"][:]
    return X[y == 0][0], X[y == 1][0]

def plot_axes(ax, data, title, time):
    for i, (axis, color) in enumerate(zip(AXES, COLORS)):
        ax.plot(time, data[:, i], label=f"{axis}-axis",
                color=color, linewidth=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Acceleration (normalized)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Plot 1: Acceleration vs Time — Walking and Jumping ────────────────────────

def plot_accel_vs_time():
    walking, jumping = load_windows()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Acceleration vs Time — Walking vs Jumping",
                 fontsize=14, fontweight="bold", y=1.01)

    plot_axes(axes[0], walking, "Walking — All Three Axes", TIME_500)
    plot_axes(axes[1], jumping, "Jumping — All Three Axes", TIME_500)

    axes[0].set_ylabel("Acceleration (normalized)")
    axes[1].set_ylabel("Acceleration (normalized)")
    axes[1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

# ── Plot 2: All 3 members — Walking overlaid ──────────────────────────────────

def plot_members_comparison():
    members = {
        "Vishwas": os.path.join(PREP_DIR, "Vishwas", "Walking", "Walk Back pocket.csv"),
        "Saurav" : os.path.join(PREP_DIR, "Saurav",  "Walking", "Saurav_walking_backpocket.csv"),
        "Tej"    : os.path.join(PREP_DIR, "Tej",     "Walking", "Tej_Walking_BackPocket.csv"),
    }
    member_colors = {"Vishwas": "#e74c3c", "Saurav": "#3498db", "Tej": "#2ecc71"}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("Walking Signal Comparison — All Members (Back Pocket)",
                 fontsize=14, fontweight="bold")

    for member, path in members.items():
        df   = pd.read_csv(path)[["x","y","z"]].iloc[:500].values
        time = np.arange(len(df)) / SAMPLING_RATE
        for i, axis in enumerate(AXES):
            axes[i].plot(time, df[:, i], label=member,
                        color=member_colors[member], linewidth=0.9, alpha=0.8)

    for i, axis in enumerate(AXES):
        axes[i].set_ylabel(f"{axis}-axis")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title("X-axis", fontsize=11)
    axes[1].set_title("Y-axis", fontsize=11)
    axes[2].set_title("Z-axis", fontsize=11)
    axes[2].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

# ── Plot 3: Raw vs Preprocessed ───────────────────────────────────────────────

def plot_raw_vs_prep():
    raw_path  = os.path.join(RAW_DIR,  "Vishwas", "Walk Back pocket.csv")
    prep_path = os.path.join(PREP_DIR, "Vishwas", "Walking", "Walk Back pocket.csv")

    raw  = pd.read_csv(raw_path).rename(columns=COL_MAP)[["x","y","z"]].iloc[:500]
    prep = pd.read_csv(prep_path)[["x","y","z"]].iloc[:500]
    time = np.arange(500) / SAMPLING_RATE

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle("Raw vs Preprocessed Signal — Vishwas Walking",
                 fontsize=14, fontweight="bold")

    for i, (axis, color) in enumerate(zip(AXES, COLORS)):
        axes[i, 0].plot(time, raw.iloc[:, i],  color=color, linewidth=0.8)
        axes[i, 1].plot(time, prep.iloc[:, i], color=color, linewidth=0.8)
        axes[i, 0].set_ylabel(f"{axis}-axis")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Raw Signal",          fontsize=11, fontweight="bold")
            axes[i, 1].set_title("Preprocessed Signal", fontsize=11, fontweight="bold")
        if i == 2:
            axes[i, 0].set_xlabel("Time (seconds)")
            axes[i, 1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

# ── Plot 4: Dataset Metadata ──────────────────────────────────────────────────

def plot_metadata():
    with h5py.File(HDF5_PATH, "r") as f:
        y_train = f["Segmented data/Train/y_train"][:]
        y_test  = f["Segmented data/Test/y_test"][:]

    all_y = np.concatenate([y_train, y_test])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset Metadata Overview", fontsize=14, fontweight="bold")

    # Class distribution
    counts = [(all_y==0).sum(), (all_y==1).sum()]
    bars   = axes[0].bar(["Walking", "Jumping"], counts,
                          color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="white")
    axes[0].set_title("Total Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of windows")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(count), ha="center", fontweight="bold")

    # Train vs Test
    split_counts = [len(y_train), len(y_test)]
    bars2 = axes[1].bar(["Train", "Test"], split_counts,
                         color=["#3498db", "#e67e22"], width=0.5, edgecolor="white")
    axes[1].set_title("Train / Test Split", fontweight="bold")
    axes[1].set_ylabel("Number of windows")
    for bar, count in zip(bars2, split_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(count), ha="center", fontweight="bold")

    # Per class per split
    x = np.arange(2)
    axes[2].bar(x - 0.2, [(y_train==0).sum(), (y_train==1).sum()],
                0.35, label="Train", color="#3498db")
    axes[2].bar(x + 0.2, [(y_test==0).sum(),  (y_test==1).sum()],
                0.35, label="Test",  color="#e67e22")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Walking", "Jumping"])
    axes[2].set_title("Class Breakdown per Split", fontweight="bold")
    axes[2].set_ylabel("Number of windows")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()

# ── Plot 5: 3D Scatter — Mean acceleration per window ─────────────────────────

def plot_3d_scatter():
    from matplotlib.lines import Line2D
    with h5py.File(HDF5_PATH, "r") as f:
        X = f["Segmented data/Train/X_train"][:]
        y = f["Segmented data/Train/y_train"][:]

    means  = X.mean(axis=1)
    colors = ["#2ecc71" if l == 0 else "#e74c3c" for l in y]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(means[:,0], means[:,1], means[:,2],
               c=colors, alpha=0.6, s=25)

    ax.set_xlabel("Mean X")
    ax.set_ylabel("Mean Y")
    ax.set_zlabel("Mean Z")
    ax.set_title("3D Scatter — Mean Acceleration per Window\n(Green=Walking, Red=Jumping)",
                 fontsize=12, fontweight="bold")

    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#2ecc71", markersize=10, label="Walking"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#e74c3c", markersize=10, label="Jumping"),
    ])
    plt.tight_layout()
    plt.show()

# ── Plot 6: Signal Heatmap — Walking vs Jumping ───────────────────────────────

def plot_heatmap():
    with h5py.File(HDF5_PATH, "r") as f:
        X = f["Segmented data/Train/X_train"][:]
        y = f["Segmented data/Train/y_train"][:]

    walking_windows = X[y == 0][:10]
    jumping_windows = X[y == 1][:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Signal Intensity Heatmap — First 10 Windows per Class",
                 fontsize=14, fontweight="bold")

    for ax, windows, title, cmap in zip(
        axes,
        [walking_windows, jumping_windows],
        ["Walking", "Jumping"],
        ["YlGn", "YlOrRd"]
    ):
        # Use magnitude (norm of x,y,z) for each sample
        magnitude = np.linalg.norm(windows, axis=2)
        im = ax.imshow(magnitude, aspect="auto", cmap=cmap,
                      extent=[0, 500/SAMPLING_RATE, 10, 0])
        ax.set_title(f"{title} — Signal Magnitude", fontweight="bold")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Window index")
        plt.colorbar(im, ax=ax, label="Magnitude")

    plt.tight_layout()
    plt.show()

# ── Plot 7: Jumping vs Walking — Side by side all 3 members ──────────────────

def plot_jumping_vs_walking_members():
    files = {
        "Vishwas": {
            "walking": os.path.join(PREP_DIR, "Vishwas", "Walking", "Walk Back pocket.csv"),
            "jumping": os.path.join(PREP_DIR, "Vishwas", "Jumping", "Jump Back Pocket.csv"),
        },
        "Saurav": {
            "walking": os.path.join(PREP_DIR, "Saurav", "Walking", "Saurav_walking_backpocket.csv"),
            "jumping": os.path.join(PREP_DIR, "Saurav", "Jumping", "Saurav_jumping_rightpocketpant.csv"),
        },
        "Tej": {
            "walking": os.path.join(PREP_DIR, "Tej", "Walking", "Tej_Walking_BackPocket.csv"),
            "jumping": os.path.join(PREP_DIR, "Tej", "Jumping", "Tej_Jumping_BackPocket.csv"),
        },
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle("Walking vs Jumping — Z-axis Signal per Member",
                 fontsize=14, fontweight="bold")

    for row, (member, actions) in enumerate(files.items()):
        for col, (action, path) in enumerate(actions.items()):
            df   = pd.read_csv(path)[["x","y","z"]].iloc[:500].values
            time = np.arange(len(df)) / SAMPLING_RATE
            color = "#2ecc71" if action == "walking" else "#e74c3c"
            axes[row, col].plot(time, df[:, 2], color=color, linewidth=0.9)
            axes[row, col].set_title(f"{member} — {action.capitalize()} (Z-axis)",
                                     fontweight="bold")
            axes[row, col].set_ylabel("Acceleration")
            axes[row, col].grid(True, alpha=0.3)
            if row == 2:
                axes[row, col].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()

# ── Run all plots ─────────────────────────────────────────────────────────────

print("Generating Plot 1: Acceleration vs Time...")
plot_accel_vs_time()

print("Generating Plot 2: Members Comparison...")
plot_members_comparison()

print("Generating Plot 3: Raw vs Preprocessed...")
plot_raw_vs_prep()

print("Generating Plot 4: Dataset Metadata...")
plot_metadata()

print("Generating Plot 5: 3D Scatter...")
plot_3d_scatter()

print("Generating Plot 6: Signal Heatmap...")
plot_heatmap()

print("Generating Plot 7: Jumping vs Walking per Member...")
plot_jumping_vs_walking_members()

print("\n✅ All plots generated!")
