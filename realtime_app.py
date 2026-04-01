import tkinter as tk
import threading
import requests
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from collections import deque
import time

# ── Config ────────────────────────────────────────────────────────────────────

PHYPHOX_URL   = "http://10.216.130.141:8080/"
MODEL_PATH    = r"D:\Vs_code\ELEC292_Project\models\model.joblib"
SCALER_PATH   = r"D:\Vs_code\ELEC292_Project\models\scaler.joblib"
WINDOW_SIZE   = 500
SAMPLING_RATE = 250
FETCH_INTERVAL = 0.01   # fetch every 25ms ≈ 40 samples/sec

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

def predict_window(window):
    signal = np.array(window)

    # Normalize
    sc     = StandardScaler()
    signal = sc.fit_transform(signal)

    # Extract features from all 3 axes
    features = []
    for axis in range(3):
        features.extend(extract_features(signal[:, axis]))

    
    features = scaler.transform([features])[0]
    label    = model.predict([features])[0]
    return "Walking" if label == 0 else "Jumping"

# ── Real-time App ─────────────────────────────────────────────────────────────

class RealTimeApp:
    def __init__(self, root):
        self.root      = root
        self.root.title("Real-Time Activity Recognizer")
        self.root.geometry("500x400")

        self.running   = False
        self.buffer    = deque(maxlen=WINDOW_SIZE)
        self.last_x    = None
        self.last_y    = None
        self.last_z    = None

        self._build_ui()

    def _build_ui(self):
        # Title
        tk.Label(self.root, text="Real-Time Activity Recognizer",
                 font=("Helvetica", 16, "bold")).pack(pady=15)

        # Activity display
        self.activity_label = tk.Label(self.root, text="--",
                                       font=("Helvetica", 48, "bold"),
                                       fg="#3498db")
        self.activity_label.pack(pady=20)

        # Buffer progress
        tk.Label(self.root, text="Buffer", font=("Helvetica", 11)).pack()
        self.progress_var = tk.DoubleVar()
        self.progress     = tk.Scale(self.root, variable=self.progress_var,
                                     from_=0, to=WINDOW_SIZE,
                                     orient=tk.HORIZONTAL, length=400,
                                     state="disabled")
        self.progress.pack(pady=5)

        self.buffer_label = tk.Label(self.root, text="0 / 2000 samples",
                                     font=("Helvetica", 10), fg="gray")
        self.buffer_label.pack()

        # Status
        self.status_label = tk.Label(self.root, text="Press Start to begin",
                                     font=("Helvetica", 11), fg="gray")
        self.status_label.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="Start",
                                   command=self.start,
                                   bg="#2ecc71", fg="white",
                                   font=("Helvetica", 12), padx=20)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="Stop",
                                  command=self.stop,
                                  bg="#e74c3c", fg="white",
                                  font=("Helvetica", 12), padx=20,
                                  state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=10)

    def start(self):
        self.running = True
        self.buffer.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Collecting data...", fg="#2ecc71")

        # Start data collection in background thread
        self.thread = threading.Thread(target=self.collect_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopped", fg="gray")
        self.activity_label.config(text="--", fg="#3498db")

    def collect_loop(self):
        while self.running:
            try:
                response = requests.get(
                    f"{PHYPHOX_URL}/get?accX&accY&accZ",
                    timeout=2
                )
                data = response.json()["buffer"]

                x = data["accX"]["buffer"][0]
                y = data["accY"]["buffer"][0]
                z = data["accZ"]["buffer"][0]

                # Only add if value changed (new sample)
                if (x, y, z) != (self.last_x, self.last_y, self.last_z):
                    self.buffer.append([x, y, z])
                    self.last_x, self.last_y, self.last_z = x, y, z

                    # Update UI
                    self.root.after(0, self.update_progress)

                    # Predict when buffer is full
                    if len(self.buffer) == WINDOW_SIZE:
                        self.root.after(0, self.run_prediction)

                time.sleep(FETCH_INTERVAL)

            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Connection error: {e}", fg="#e74c3c"))
                time.sleep(1)

    def update_progress(self):
        count = len(self.buffer)
        self.progress_var.set(count)
        self.buffer_label.config(text=f"{count} / {WINDOW_SIZE} samples")

    def run_prediction(self):
        self.status_label.config(text="Predicting...", fg="#e67e22")
        self.root.update()

        result = predict_window(list(self.buffer))

        color  = "#2ecc71" if result == "Walking" else "#e74c3c"
        self.activity_label.config(text=result, fg=color)
        self.status_label.config(text="Collecting next window...", fg="#3498db")

        # Clear buffer for next window
        self.buffer.clear()

# ── Run ───────────────────────────────────────────────────────────────────────

root = tk.Tk()
RealTimeApp(root)
root.mainloop()