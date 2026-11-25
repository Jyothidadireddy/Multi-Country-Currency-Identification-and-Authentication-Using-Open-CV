"""
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING SCRIPT — Image-Based Currency Authenticity Model      ║
║  Loads US/UK real/fake images → extracts features → trains      ║
║  Saves: country classifier + auth classifier + pipeline (.pkl)  ║
╚══════════════════════════════════════════════════════════════════╝

Dataset: dataset/US/{real,fake}  +  dataset/UK/{real,fake}
Features: Color histograms, texture, edges, sharpness, wavelet stats
Classes:
  Country: 0 = US, 1 = UK
  Auth:    0 = Real, 1 = Fake

Usage:  python train_image_model.py
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import cv2
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224

# Output paths
COUNTRY_MODEL_PATH = os.path.join(MODEL_DIR, "image_country_clf.pkl")
AUTH_MODEL_PATH = os.path.join(MODEL_DIR, "image_auth_clf.pkl")
PIPELINE_PATH = os.path.join(MODEL_DIR, "image_feature_pipeline.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "image_training_report.json")

# Countries to train on
COUNTRIES = {
    0: {"code": "USD", "name": "US Dollar", "folder": "US"},
    1: {"code": "GBP", "name": "British Pound", "folder": "UK"},
}


def banner(text):
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}")


# ═══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_image_features(image):
    """
    Extract a rich feature vector from a currency image.

    Features (total ~130 dimensions):
    - Color histogram (HSV): 48 bins
    - Color statistics (mean, std per channel): 6
    - Grayscale histogram: 32 bins
    - Texture (LBP-like): 10
    - Edge features: 4
    - Wavelet-like stats: 8
    - Sharpness & quality: 4
    - GLCM-like features: 4
    """
    if image is None:
        return None

    # Resize to standard size
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    features = []

    # ── 1. HSV Color Histogram (48 values) ───────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch, bins in [(0, 18), (1, 16), (2, 14)]:
        hist = cv2.calcHist([hsv], [ch], None, [bins],
                            [0, 180] if ch == 0 else [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        features.extend(hist)

    # ── 2. Color Statistics (6 values) ───────────────────────────
    for ch in range(3):
        channel = img[:, :, ch].astype(np.float64)
        features.append(np.mean(channel) / 255.0)
        features.append(np.std(channel) / 255.0)

    # ── 3. Grayscale Histogram (32 values) ───────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    gray_hist = gray_hist.flatten() / (gray_hist.sum() + 1e-10)
    features.extend(gray_hist)

    # ── 4. Texture Features (LBP-like, 10 values) ───────────────
    gray_f = gray.astype(np.float64)
    # Compute local binary pattern approximation using gradients
    dx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)

    # Magnitude histogram (5 bins)
    mag_hist = np.histogram(magnitude, bins=5, range=(0, 500))[0]
    mag_hist = mag_hist.astype(np.float64) / (mag_hist.sum() + 1e-10)
    features.extend(mag_hist)

    # Direction histogram (5 bins)
    dir_hist = np.histogram(direction, bins=5, range=(-np.pi, np.pi))[0]
    dir_hist = dir_hist.astype(np.float64) / (dir_hist.sum() + 1e-10)
    features.extend(dir_hist)

    # ── 5. Edge Features (4 values) ──────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    laplacian = cv2.Laplacian(gray_f, cv2.CV_64F)
    features.append(np.var(laplacian) / 10000.0)  # normalised sharpness
    features.append(np.mean(np.abs(laplacian)) / 100.0)
    features.append(np.max(np.abs(laplacian)) / 1000.0)

    # ── 6. Wavelet-like Stats (8 values) ─────────────────────────
    h_detail = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    v_detail = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)

    for arr in [h_detail, v_detail]:
        features.append(np.var(arr) / 10000.0)
        features.append(float(np.mean(np.abs(arr))) / 100.0)
        from scipy import stats as sp_stats
        features.append(float(sp_stats.skew(arr.ravel())))
        features.append(float(sp_stats.kurtosis(arr.ravel())))

    # ── 7. Quality Features (4 values) ───────────────────────────
    # Contrast
    features.append(np.std(gray_f) / 128.0)
    # Brightness
    features.append(np.mean(gray_f) / 255.0)
    # Colour uniformity
    features.append(np.mean([np.std(img[:, :, c]) for c in range(3)]) / 128.0)
    # Unique intensity levels
    features.append(len(np.unique(gray)) / 256.0)

    # ── 8. GLCM-like Features (4 values) ─────────────────────────
    # Approximate using co-occurrence differences
    gray_small = cv2.resize(gray, (64, 64))
    h_diff = np.abs(gray_small[:, 1:].astype(np.int16) - gray_small[:, :-1].astype(np.int16))
    v_diff = np.abs(gray_small[1:, :].astype(np.int16) - gray_small[:-1, :].astype(np.int16))
    features.append(np.mean(h_diff) / 128.0)
    features.append(np.std(h_diff) / 128.0)
    features.append(np.mean(v_diff) / 128.0)
    features.append(np.std(v_diff) / 128.0)

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_dataset():
    """Load images from dataset/US/{real,fake} and dataset/UK/{real,fake}."""
    banner("STEP 1: Loading Dataset")

    images = []
    country_labels = []
    auth_labels = []
    file_paths = []

    for country_id, info in COUNTRIES.items():
        folder = os.path.join(DATASET_DIR, info["folder"])
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            img_dir = os.path.join(folder, label_name)
            if not os.path.exists(img_dir):
                print(f"  [!] Missing: {img_dir}")
                continue

            files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            for fname in files:
                fpath = os.path.join(img_dir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                country_labels.append(country_id)
                auth_labels.append(label_val)
                file_paths.append(fpath)

        real_count = sum(1 for a, c in zip(auth_labels, country_labels)
                         if c == country_id and a == 0)
        fake_count = sum(1 for a, c in zip(auth_labels, country_labels)
                         if c == country_id and a == 1)
        print(f"  [OK] {info['code']}: {real_count} real + {fake_count} fake")

    print(f"\n  Total: {len(images)} images")
    print(f"  Real: {sum(1 for a in auth_labels if a == 0)}")
    print(f"  Fake: {sum(1 for a in auth_labels if a == 1)}")

    return images, np.array(country_labels), np.array(auth_labels)


# ═══════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════
def augment_dataset(images, country_labels, auth_labels, multiplier=4):
    """Create augmented copies to increase training set size."""
    banner("STEP 2: Augmenting Dataset")

    aug_images = list(images)
    aug_country = list(country_labels)
    aug_auth = list(auth_labels)

    for img, yc, ya in zip(images, country_labels, auth_labels):
        for _ in range(multiplier):
            aug = img.copy()

            # Random horizontal flip
            if np.random.random() > 0.5:
                aug = cv2.flip(aug, 1)

            # Random brightness/contrast
            alpha = np.random.uniform(0.75, 1.25)
            beta = np.random.randint(-20, 20)
            aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)

            # Random rotation (-8 to 8 degrees)
            angle = np.random.uniform(-8, 8)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random slight crop & resize
            if np.random.random() > 0.4:
                crop = np.random.randint(3, 15)
                aug = aug[crop:h - crop, crop:w - crop]
                aug = cv2.resize(aug, (IMG_SIZE, IMG_SIZE))

            # Random noise
            if np.random.random() > 0.6:
                noise = np.random.normal(0, 8, aug.shape).astype(np.int16)
                aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random colour jitter
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.randint(-8, 8), 0, 179)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(-20, 20), 0, 255)
                aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            aug_images.append(aug)
            aug_country.append(yc)
            aug_auth.append(ya)

    print(f"  Original: {len(images)} -> Augmented: {len(aug_images)} images")
    print(f"  Multiplier: {multiplier}x + originals")

    return aug_images, np.array(aug_country), np.array(aug_auth)


# ═══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (batch)
# ═══════════════════════════════════════════════════════════════════
def extract_all_features(images):
    """Extract features from all images."""
    banner("STEP 3: Extracting Features")

    features = []
    for i, img in enumerate(images):
        feat = extract_image_features(img)
        if feat is not None:
            features.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")

    X = np.array(features)
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Features per image: {X.shape[1]}")

    return X


# ═══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_models(X, y_country, y_auth):
    """Train country classifier + authenticity classifier."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   VotingClassifier)
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, classification_report, confusion_matrix)
    from sklearn.pipeline import Pipeline

    banner("STEP 4: Training Models")

    # ── Split ────────────────────────────────────────────────────
    X_train, X_test, yc_train, yc_test, ya_train, ya_test = train_test_split(
        X, y_country, y_auth, test_size=0.2, random_state=42, stratify=y_auth
    )
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")

    # ── Scale ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ===================================================================
    #  COUNTRY CLASSIFIER (US vs UK)
    # ===================================================================
    print("\n  -- Training Country Classifier (US vs UK) --")

    country_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
    }

    best_country_name = None
    best_country_acc = 0
    best_country_model = None

    print(f"\n  {'Model':<22} {'Accuracy':>9} {'F1':>8} {'CV(5)':>8}")
    print(f"  {'-' * 50}")

    for name, model in country_models.items():
        model.fit(X_train_s, yc_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(yc_test, y_pred)
        f1 = f1_score(yc_test, y_pred, average="weighted", zero_division=0)
        cv = cross_val_score(model, X_train_s, yc_train, cv=5).mean()
        print(f"  {name:<22} {acc:>8.1%} {f1:>7.1%} {cv:>7.1%}")
        if acc > best_country_acc:
            best_country_acc = acc
            best_country_name = name
            best_country_model = model

    # Ensemble
    country_ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
            ("svm", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ],
        voting="soft"
    )
    country_ensemble.fit(X_train_s, yc_train)
    ens_acc = accuracy_score(yc_test, country_ensemble.predict(X_test_s))
    ens_f1 = f1_score(yc_test, country_ensemble.predict(X_test_s), average="weighted")
    ens_cv = cross_val_score(country_ensemble, X_train_s, yc_train, cv=5).mean()
    print(f"  {'Ensemble (Voting)':<22} {ens_acc:>8.1%} {ens_f1:>7.1%} {ens_cv:>7.1%}")

    if ens_acc >= best_country_acc:
        best_country_acc = ens_acc
        best_country_name = "Ensemble (Voting)"
        best_country_model = country_ensemble

    print(f"\n  * Best Country Model: {best_country_name} ({best_country_acc:.1%})")

    # ===================================================================
    #  AUTHENTICITY CLASSIFIER (Real vs Fake)
    # ===================================================================
    print("\n  -- Training Authenticity Classifier (Real vs Fake) --")

    auth_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
    }

    best_auth_name = None
    best_auth_acc = 0
    best_auth_model = None

    print(f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV(5)':>8}")
    print(f"  {'-' * 68}")

    for name, model in auth_models.items():
        model.fit(X_train_s, ya_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(ya_test, y_pred)
        prec = precision_score(ya_test, y_pred, zero_division=0)
        rec = recall_score(ya_test, y_pred, zero_division=0)
        f1 = f1_score(ya_test, y_pred, zero_division=0)
        cv = cross_val_score(model, X_train_s, ya_train, cv=5).mean()
        print(f"  {name:<22} {acc:>8.1%} {prec:>9.1%} {rec:>8.1%} {f1:>7.1%} {cv:>7.1%}")
        if acc > best_auth_acc:
            best_auth_acc = acc
            best_auth_name = name
            best_auth_model = model

    # Ensemble
    auth_ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)),
            ("svm", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ],
        voting="soft"
    )
    auth_ensemble.fit(X_train_s, ya_train)
    ens_pred = auth_ensemble.predict(X_test_s)
    ens_acc = accuracy_score(ya_test, ens_pred)
    ens_prec = precision_score(ya_test, ens_pred, zero_division=0)
    ens_rec = recall_score(ya_test, ens_pred, zero_division=0)
    ens_f1 = f1_score(ya_test, ens_pred, zero_division=0)
    ens_cv = cross_val_score(auth_ensemble, X_train_s, ya_train, cv=5).mean()
    print(f"  {'Ensemble (Voting)':<22} {ens_acc:>8.1%} {ens_prec:>9.1%} "
          f"{ens_rec:>8.1%} {ens_f1:>7.1%} {ens_cv:>7.1%}")

    if ens_acc >= best_auth_acc:
        best_auth_acc = ens_acc
        best_auth_name = "Ensemble (Voting)"
        best_auth_model = auth_ensemble

    print(f"\n  * Best Auth Model: {best_auth_name} ({best_auth_acc:.1%})")

    # -- Detailed reports -----------------------------------------
    print("\n  -- Authenticity Classification Report --")
    y_pred_auth = best_auth_model.predict(X_test_s)
    print(classification_report(ya_test, y_pred_auth,
                                 target_names=["REAL", "FAKE"]))

    cm = confusion_matrix(ya_test, y_pred_auth)
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                REAL   FAKE")
    print(f"  Actual REAL   {cm[0][0]:>4}   {cm[0][1]:>4}")
    print(f"  Actual FAKE   {cm[1][0]:>4}   {cm[1][1]:>4}")

    results = {
        "country_model": best_country_name,
        "country_accuracy": round(best_country_acc, 4),
        "auth_model": best_auth_name,
        "auth_accuracy": round(best_auth_acc, 4),
    }

    return best_country_model, best_auth_model, scaler, results


# ===================================================================
#  SAVE MODELS
# ===================================================================
def save_models(country_model, auth_model, scaler, results):
    """Save trained models and pipeline to disk."""
    banner("STEP 5: Saving Models")

    # Save country classifier
    with open(COUNTRY_MODEL_PATH, "wb") as f:
        pickle.dump(country_model, f)
    print(f"  [OK] Country model saved: {COUNTRY_MODEL_PATH}")

    # Save authenticity classifier
    with open(AUTH_MODEL_PATH, "wb") as f:
        pickle.dump(auth_model, f)
    print(f"  [OK] Auth model saved: {AUTH_MODEL_PATH}")

    # Save feature pipeline (scaler + country mapping)
    pipeline = {
        "scaler": scaler,
        "countries": COUNTRIES,
        "img_size": IMG_SIZE,
    }
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  [OK] Pipeline saved: {PIPELINE_PATH}")

    # Save training report
    report = {
        **results,
        "dataset": "US + UK (real/fake images)",
        "features": "Color histogram + texture + edges + wavelet + quality + GLCM",
        "countries": {str(k): v for k, v in COUNTRIES.items()},
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] Report saved: {REPORT_PATH}")

    # File sizes
    for path in [COUNTRY_MODEL_PATH, AUTH_MODEL_PATH, PIPELINE_PATH]:
        size = os.path.getsize(path) / 1024
        print(f"    {os.path.basename(path)}: {size:.1f} KB")


# ===================================================================
#  MAIN
# ===================================================================
def main():
    print("\n" + "=" * 60)
    print("  [*]  IMAGE-BASED CURRENCY AUTHENTICITY MODEL")
    print("  US + UK Datasets - Feature Extraction - Ensemble ML")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Load
    images, y_country, y_auth = load_dataset()
    if len(images) == 0:
        print("\n  [X] No images found! Check dataset/US and dataset/UK folders.")
        sys.exit(1)

    # Step 2: Augment
    images, y_country, y_auth = augment_dataset(images, y_country, y_auth, multiplier=4)

    # Step 3: Extract features
    X = extract_all_features(images)

    # Step 4: Train
    country_model, auth_model, scaler, results = train_models(X, y_country, y_auth)

    # Step 5: Save
    save_models(country_model, auth_model, scaler, results)

    elapsed = time.time() - t_start
    banner(f"DONE -- Total time: {elapsed:.1f}s")
    print(f"\n  [OK] Image models are ready for upload-based detection!")
    print(f"  -> Run: python app.py")
    print(f"  -> Click 'Other Country' button to use image upload\n")


if __name__ == "__main__":
    main()
