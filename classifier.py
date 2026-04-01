import numpy as np
import matplotlib.pyplot as plt
import h5py
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ── Config ────────────────────────────────────────────────────────────────────

HDF5_PATH = r"D:\Vs_code\ELEC292_Project\dataset.h5"
MODEL_DIR = Path(__file__).resolve().parent / "models"

# ── Load features ─────────────────────────────────────────────────────────────

def load_features():
    with h5py.File(HDF5_PATH, "r") as f:
        F_train = f["Features/F_train"][:]
        F_test  = f["Features/F_test"][:]
        y_train = f["Features/y_train"][:]
        y_test  = f["Features/y_test"][:]
    return F_train, F_test, y_train, y_test

# ── Train logistic regression ─────────────────────────────────────────────────

def train(F_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(F_train, y_train)
    return model

# ── Evaluate on test set ──────────────────────────────────────────────────────

def evaluate(model, F_test, y_test):
    y_pred   = model.predict(F_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"  ✅ Test Accuracy : {accuracy:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  True Walking  correct : {cm[0,0]}  |  misclassified as jumping : {cm[0,1]}")
    print(f"  True Jumping  correct : {cm[1,1]}  |  misclassified as walking : {cm[1,0]}")

    return y_pred, cm

# ── Plot training curve ───────────────────────────────────────────────────────

def plot_training_curve(model, F_train, y_train):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_acc, test_acc = [], []

    F_test_curve, y_test_curve = load_features()[1], load_features()[3]

    for size in train_sizes:
        n     = max(1, int(len(F_train) * size))
        m     = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        m.fit(F_train[:n], y_train[:n])
        train_acc.append(accuracy_score(y_train[:n], m.predict(F_train[:n])) * 100)
        test_acc.append(accuracy_score(y_test_curve, m.predict(F_test_curve)) * 100)

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes * 100, train_acc, label="Train Accuracy", color="#3498db", marker="o")
    plt.plot(train_sizes * 100, test_acc,  label="Test Accuracy",  color="#e74c3c", marker="o")
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Curve — Logistic Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ── Plot confusion matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Walking", "Jumping"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.show()

# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    F_train, F_test, y_train, y_test = load_features()

    print("🏋️  Training logistic regression...")
    model = train(F_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"  💾 Model saved to {MODEL_DIR / 'model.joblib'}")

    print("\n📊 Evaluating on test set...")
    y_pred, cm = evaluate(model, F_test, y_test)

    plot_training_curve(model, F_train, y_train)
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()
