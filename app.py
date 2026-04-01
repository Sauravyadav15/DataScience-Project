import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
from scipy.stats import skew, kurtosis
import os

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = r"D:\Vs_code\ELEC292_Project\models\model.joblib"
SCALER_PATH   = r"D:\Vs_code\ELEC292_Project\models\scaler.joblib"
WINDOW_SIZE   = 2000
STEP_SIZE     = 1000
SAMPLING_RATE = 400

COL_MAP = {
    "Acceleration x (m/s^2)":        "x",
    "Acceleration y (m/s^2)":        "y",
    "Acceleration z (m/s^2)":        "z",
    "Linear Acceleration x (m/s^2)": "x",
    "Linear Acceleration y (m/s^2)": "y",
    "Linear Acceleration z (m/s^2)": "z",
}

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df):
    df = df.rename(columns=COL_MAP)[["x", "y", "z"]].astype(float)
    df = df.interpolate(method="linear", limit=5).dropna().reset_index(drop=True)
    for col in ["x", "y", "z"]:
        mean, std = df[col].mean(), df[col].std()
        df[col]   = df[col].clip(mean - 3*std, mean + 3*std)
    from sklearn.preprocessing import StandardScaler
    df[["x","y","z"]] = StandardScaler().fit_transform(df[["x","y","z"]])
    return df.values

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(signal):
    fts = [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        np.max(signal) - np.min(signal), np.median(signal), np.var(signal),
        skew(signal), kurtosis(signal), np.sqrt(np.mean(signal**2)),
        np.sum(np.diff(np.sign(np.diff(signal))) < 0),
        np.sum(signal**2),
        np.sum(np.diff(np.sign(signal)) != 0),
        np.percentile(signal, 25), np.percentile(signal, 75),
    ]
    fft_vals         = np.abs(np.fft.fft(signal))[:len(signal)//2]
    freqs            = np.fft.fftfreq(len(signal), d=1/SAMPLING_RATE)[:len(signal)//2]
    psd              = fft_vals**2 / (np.sum(fft_vals**2) + 1e-10)
    fts.extend([
        freqs[np.argmax(fft_vals)],
        np.sum(fft_vals**2),
        -np.sum(psd * np.log2(psd + 1e-10)),
    ])
    return fts

def extract_window_features(window):
    features = []
    for axis in range(3):
        features.extend(extract_features(window[:, axis]))
    return features

# ── Predict ───────────────────────────────────────────────────────────────────

def predict(signal):
    windows, starts = [], []
    n = (len(signal) - WINDOW_SIZE) // STEP_SIZE + 1
    for i in range(n):
        start = i * STEP_SIZE
        windows.append(signal[start : start + WINDOW_SIZE])
        starts.append(start)

    features = np.array([extract_window_features(w) for w in windows])
    features = scaler.transform(features)
    labels   = model.predict(features)
    return windows, starts, labels

# ── Plot results ──────────────────────────────────────────────────────────────

def plot_results(signal, starts, labels, canvas_frame):
    fig, ax = plt.subplots(figsize=(10, 4))
    time    = np.arange(len(signal)) / SAMPLING_RATE

    ax.plot(time, signal[:, 0], color="gray", linewidth=0.6, alpha=0.5, label="X-axis")

    for start, label in zip(starts, labels):
        end   = start + WINDOW_SIZE
        color = "#2ecc71" if label == 0 else "#e74c3c"
        ax.axvspan(time[start], time[min(end, len(signal)-1)],
                   alpha=0.3, color=color)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ecc71", alpha=0.5, label="Walking"),
        Patch(color="#e74c3c", alpha=0.5, label="Jumping"),
    ])
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration")
    ax.set_title("Activity Prediction per Window")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for widget in canvas_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ── Save output CSV ───────────────────────────────────────────────────────────

def save_output(starts, labels, output_path):
    df = pd.DataFrame({
        "window_start_sample" : starts,
        "window_start_time_s" : [s / SAMPLING_RATE for s in starts],
        "label"               : ["walking" if l == 0 else "jumping" for l in labels],
    })
    df.to_csv(output_path, index=False)

# ── Main App ──────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root       = root
        self.root.title("Activity Recognizer")
        self.root.geometry("900x650")
        self.input_path = None

        # Top bar
        top = tk.Frame(root, pady=10)
        top.pack(fill=tk.X, padx=20)

        tk.Label(top, text="Activity Recognition App",
                 font=("Helvetica", 16, "bold")).pack(side=tk.LEFT)

        tk.Button(top, text="Browse CSV", command=self.browse,
                  bg="#3498db", fg="white", padx=10).pack(side=tk.RIGHT)

        tk.Button(top, text="Run Prediction", command=self.run,
                  bg="#2ecc71", fg="white", padx=10).pack(side=tk.RIGHT, padx=5)

        tk.Button(top, text="Save Output CSV", command=self.save,
                  bg="#e67e22", fg="white", padx=10).pack(side=tk.RIGHT, padx=5)

        # File label
        self.file_label = tk.Label(root, text="No file selected",
                                   fg="gray", font=("Helvetica", 10))
        self.file_label.pack(padx=20, anchor="w")

        # Status
        self.status = tk.Label(root, text="", fg="#2ecc71",
                               font=("Helvetica", 11))
        self.status.pack(padx=20, anchor="w")

        # Canvas for plot
        self.canvas_frame = tk.Frame(root, bg="white")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.labels  = None
        self.starts  = None

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.input_path = path
            self.file_label.config(text=f"Selected: {path}", fg="black")
            self.status.config(text="")

    def run(self):
        if not self.input_path:
            messagebox.showwarning("No file", "Please select a CSV file first!")
            return
        try:
            self.status.config(text="Processing...")
            self.root.update()

            df              = pd.read_csv(self.input_path)
            signal          = preprocess(df)
            windows, starts, labels = predict(signal)

            self.starts = starts
            self.labels = labels

            plot_results(signal, starts, labels, self.canvas_frame)

            walking = (labels == 0).sum()
            jumping = (labels == 1).sum()
            self.status.config(
                text=f"✅ Done!  {len(labels)} windows  |  Walking: {walking}  |  Jumping: {jumping}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="❌ Error occurred")

    def save(self):
        if self.labels is None:
            messagebox.showwarning("No results", "Run prediction first!")
            return

        # Auto-generate filename: outputdata1.csv, outputdata2.csv, ...
        output_dir = r"D:\Vs_code\ELEC292_Project\outputs"
        os.makedirs(output_dir, exist_ok=True)

        counter = 1
        while os.path.exists(os.path.join(output_dir, f"outputdata{counter}.csv")):
            counter += 1

        output_path = os.path.join(output_dir, f"outputdata{counter}.csv")
        save_output(self.starts, self.labels, output_path)

        messagebox.showinfo("Saved", f"Output saved as:\noutputdata{counter}.csv")
        self.status.config(text=f"💾 Saved → outputdata{counter}.csv")

# ── Run ───────────────────────────────────────────────────────────────────────

root = tk.Tk()
App(root)
root.mainloop()