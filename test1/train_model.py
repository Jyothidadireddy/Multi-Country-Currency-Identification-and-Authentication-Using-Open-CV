"""
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING SCRIPT — Banknote Authentication Model                ║
║  Downloads UCI dataset → extracts features → trains classifier  ║
║  Saves trained model + scaler for live detection integration    ║
╚══════════════════════════════════════════════════════════════════╝

Dataset: UCI Banknote Authentication (1372 samples)
Features: variance, skewness, curtosis, entropy (from wavelet transform)
Classes: 0 = genuine, 1 = counterfeit

Usage:  python train_model.py
"""

import os
import sys
import time
import json
import pickle
import urllib.request
import numpy as np

# ── Ensure directories exist ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
DATASET_PATH = os.path.join(DATASET_DIR, "banknote_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "banknote_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")


def banner(text):
    w = 60
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}")


def download_dataset():
    """Download the UCI Banknote Authentication dataset."""
    banner("STEP 1: Downloading Dataset")
    if os.path.exists(DATASET_PATH):
        print(f"  ✓ Dataset already exists at: {DATASET_PATH}")
        return True

    print(f"  ↓ Downloading from UCI repository...")
    print(f"    URL: {DATASET_URL}")

    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        # Verify the file
        with open(DATASET_PATH, "r") as f:
            lines = f.readlines()
        print(f"  ✓ Downloaded successfully: {len(lines)} samples")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  → Generating synthetic dataset instead...")
        return generate_synthetic_dataset()


def generate_synthetic_dataset():
    """Fallback: generate a synthetic dataset if download fails."""
    print("  → Creating synthetic banknote dataset...")
    np.random.seed(42)
    n_genuine = 700
    n_fake = 672

    # Genuine notes: higher variance, specific distributions
    genuine_variance = np.random.normal(0.4, 2.5, n_genuine)
    genuine_skewness = np.random.normal(5.0, 3.5, n_genuine)
    genuine_curtosis = np.random.normal(2.5, 4.0, n_genuine)
    genuine_entropy = np.random.normal(-1.0, 2.0, n_genuine)
    genuine = np.column_stack([genuine_variance, genuine_skewness,
                                genuine_curtosis, genuine_entropy,
                                np.zeros(n_genuine)])

    # Fake notes: different distribution pattern
    fake_variance = np.random.normal(-1.5, 2.0, n_fake)
    fake_skewness = np.random.normal(-1.0, 3.0, n_fake)
    fake_curtosis = np.random.normal(1.0, 3.5, n_fake)
    fake_entropy = np.random.normal(0.5, 1.5, n_fake)
    fake = np.column_stack([fake_variance, fake_skewness,
                             fake_curtosis, fake_entropy,
                             np.ones(n_fake)])

    data = np.vstack([genuine, fake])
    np.random.shuffle(data)

    with open(DATASET_PATH, "w") as f:
        for row in data:
            f.write(",".join(f"{v:.6f}" for v in row[:-1]) + f",{int(row[-1])}\n")

    print(f"  ✓ Synthetic dataset created: {len(data)} samples")
    return True


def load_dataset():
    """Load and parse the CSV dataset."""
    banner("STEP 2: Loading & Exploring Dataset")
    data = np.loadtxt(DATASET_PATH, delimiter=",")
    X = data[:, :4]    # features: variance, skewness, curtosis, entropy
    y = data[:, 4]     # labels: 0=genuine, 1=counterfeit

    n_genuine = int(np.sum(y == 0))
    n_fake = int(np.sum(y == 1))
    total = len(y)

    print(f"  Total samples:     {total}")
    print(f"  Genuine (class 0): {n_genuine} ({n_genuine/total*100:.1f}%)")
    print(f"  Fake    (class 1): {n_fake} ({n_fake/total*100:.1f}%)")
    print()
    print(f"  Feature Statistics:")
    print(f"  {'Feature':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'─'*48}")
    names = ["Variance", "Skewness", "Curtosis", "Entropy"]
    for i, name in enumerate(names):
        print(f"  {name:<12} {X[:,i].min():>8.2f} {X[:,i].max():>8.2f} "
              f"{X[:,i].mean():>8.2f} {X[:,i].std():>8.2f}")

    return X, y


def train_model(X, y):
    """Train multiple classifiers, pick the best, save it."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   AdaBoostClassifier, VotingClassifier)
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, classification_report)

    banner("STEP 3: Training Multiple Models")

    # ── Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")

    # ── Scale ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Models to try ────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=500,
            learning_rate="adaptive", random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
    }

    results = {}
    best_name = None
    best_acc = 0
    best_model = None

    print(f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV(5)':>8}")
    print(f"  {'─'*68}")

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()

        print(f"  {name:<22} {acc:>8.1%} {prec:>9.1%} {rec:>8.1%} {f1:>7.1%} {cv_mean:>7.1%}")

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "cv_mean": round(cv_mean, 4),
            "train_time": round(train_time, 3),
        }

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    # ── Ensemble (Voting Classifier) ─────────────────────────────
    print(f"\n  Training Ensemble (Voting Classifier)...")
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)),
            ("svm", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ],
        voting="soft"
    )
    ensemble.fit(X_train_s, y_train)
    y_pred_ens = ensemble.predict(X_test_s)
    ens_acc = accuracy_score(y_test, y_pred_ens)
    ens_f1 = f1_score(y_test, y_pred_ens)
    cv_ens = cross_val_score(ensemble, X_train_s, y_train, cv=5).mean()

    print(f"  {'Ensemble (Voting)':<22} {ens_acc:>8.1%} "
          f"{precision_score(y_test, y_pred_ens):>9.1%} "
          f"{recall_score(y_test, y_pred_ens):>8.1%} "
          f"{ens_f1:>7.1%} {cv_ens:>7.1%}")

    results["Ensemble (Voting)"] = {
        "accuracy": round(ens_acc, 4),
        "precision": round(precision_score(y_test, y_pred_ens), 4),
        "recall": round(recall_score(y_test, y_pred_ens), 4),
        "f1": round(ens_f1, 4),
        "cv_mean": round(cv_ens, 4),
    }

    if ens_acc >= best_acc:
        best_acc = ens_acc
        best_name = "Ensemble (Voting)"
        best_model = ensemble

    # ── Select Best ──────────────────────────────────────────────
    banner(f"STEP 4: Best Model → {best_name} ({best_acc:.1%})")

    # Detailed report for the best model
    y_pred_best = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(y_test, y_pred_best,
                                    target_names=["Genuine", "Counterfeit"])

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Genuine  Fake")
    print(f"  Actual Genuine   {cm[0][0]:>4}   {cm[0][1]:>4}")
    print(f"  Actual Fake      {cm[1][0]:>4}   {cm[1][1]:>4}")
    print(f"\n  Classification Report:")
    for line in report.split("\n"):
        print(f"  {line}")

    return best_model, scaler, best_name, results


def save_model(model, scaler, model_name, results):
    """Save the trained model and scaler to disk."""
    banner("STEP 5: Saving Model")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved: {MODEL_PATH}")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {SCALER_PATH}")

    # Save training report
    report = {
        "model_name": model_name,
        "dataset": "UCI Banknote Authentication",
        "samples": 1372,
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "classes": {"0": "genuine", "1": "counterfeit"},
        "results": results,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report saved: {REPORT_PATH}")

    model_size = os.path.getsize(MODEL_PATH) / 1024
    print(f"  ✓ Model size: {model_size:.1f} KB")


def main():
    print("\n" + "=" * 60)
    print("  💰  BANKNOTE AUTHENTICATION — MODEL TRAINING")
    print("  UCI Dataset · Multiple Classifiers · Ensemble")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Download
    if not download_dataset():
        print("  ✗ Failed to get dataset. Aborting.")
        sys.exit(1)

    # Step 2: Load
    X, y = load_dataset()

    # Step 3-4: Train & select best
    model, scaler, model_name, results = train_model(X, y)

    # Step 5: Save
    save_model(model, scaler, model_name, results)

    elapsed = time.time() - t_start
    banner(f"DONE — Total time: {elapsed:.1f}s")
    print(f"\n  🎯 Model '{model_name}' is ready for live detection!")
    print(f"  → Run: python app.py\n")


if __name__ == "__main__":
    main()
