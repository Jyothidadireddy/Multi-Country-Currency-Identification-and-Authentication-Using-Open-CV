"""
╔══════════════════════════════════════════════════════════════════╗
║  Currency CNN Training Pipeline                                ║
║  Trains a CNN classifier for multi-currency fake detection     ║
║  Supports: 17 countries × (real / fake) classification         ║
║  Outputs: .h5 (Keras) + .pkl (sklearn backup)                  ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  - Conv2D layers with BatchNorm and MaxPooling
  - Dense classifier head
  - Two outputs: Country (17-class) + Authenticity (real/fake)

Usage:
  python train_currency_model.py
"""

import os
import sys
import json
import pickle
import numpy as np
import cv2
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "currency_images")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

CURRENCY_CODES = [
    "INR", "USD", "EUR", "GBP", "NPR", "JPY", "CNY",
    "CAD", "AUD", "RUB", "BRL", "KRW", "SAR", "AED",
    "THB", "LKR", "BDT"
]


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_dataset():
    """Load all images from dataset/currency_images/{code}/{real|fake}/."""
    images = []
    country_labels = []   # 0-16 (which currency)
    auth_labels = []      # 0 = real, 1 = fake

    print("\n[*] Loading dataset...")
    for idx, code in enumerate(CURRENCY_CODES):
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            folder = os.path.join(DATASET_DIR, code, label_name)
            if not os.path.exists(folder):
                print(f"  [!] Missing: {folder}")
                continue

            files = [f for f in os.listdir(folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            for fname in files:
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0  # normalise
                images.append(img)
                country_labels.append(idx)
                auth_labels.append(label_val)

        count_real = sum(1 for a, c in zip(auth_labels, country_labels)
                         if c == idx and a == 0)
        count_fake = sum(1 for a, c in zip(auth_labels, country_labels)
                         if c == idx and a == 1)
        print(f"  [OK] {code}: {count_real} real + {count_fake} fake")

    X = np.array(images)
    y_country = np.array(country_labels)
    y_auth = np.array(auth_labels)

    print(f"\n  Total: {len(X)} images")
    print(f"  Real: {np.sum(y_auth == 0)}, Fake: {np.sum(y_auth == 1)}")
    return X, y_country, y_auth


# ═══════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════
def augment_batch(images, labels_country, labels_auth):
    """Simple augmentation: flip, rotate, brightness."""
    aug_images = []
    aug_country = []
    aug_auth = []

    for img, yc, ya in zip(images, labels_country, labels_auth):
        aug_images.append(img)
        aug_country.append(yc)
        aug_auth.append(ya)

        # Horizontal flip
        flipped = cv2.flip(img, 1)
        aug_images.append(flipped)
        aug_country.append(yc)
        aug_auth.append(ya)

        # Slight brightness
        bright = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1).astype(np.float32)
        aug_images.append(bright)
        aug_country.append(yc)
        aug_auth.append(ya)

    return np.array(aug_images), np.array(aug_country), np.array(aug_auth)


# ═══════════════════════════════════════════════════════════════════
#  BUILD MODEL (TensorFlow/Keras CNN)
# ═══════════════════════════════════════════════════════════════════
def build_keras_model(num_countries=17):
    """Build a dual-output CNN: country classification + authenticity."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("[X] TensorFlow not installed. Installing...")
        os.system(f"{sys.executable} -m pip install tensorflow --quiet")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

    # ── Input ────────────────────────────────────────────────────
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # ── Feature Extractor (lightweight CNN) ──────────────────────
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # ── Shared dense layer ───────────────────────────────────────
    shared = layers.Dense(256, activation="relu", name="shared_dense")(x)
    shared = layers.Dropout(0.3)(shared)

    # ── Country classification head ──────────────────────────────
    country_head = layers.Dense(128, activation="relu", name="country_dense")(shared)
    country_out = layers.Dense(num_countries, activation="softmax",
                                name="country_output")(country_head)

    # ── Authenticity head (real vs fake) ─────────────────────────
    auth_head = layers.Dense(64, activation="relu", name="auth_dense")(shared)
    auth_out = layers.Dense(1, activation="sigmoid", name="auth_output")(auth_head)

    # ── Compile ──────────────────────────────────────────────────
    model = keras.Model(inputs=inputs,
                        outputs=[country_out, auth_out],
                        name="CurrencyDetectorCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "country_output": "sparse_categorical_crossentropy",
            "auth_output": "binary_crossentropy",
        },
        loss_weights={"country_output": 1.0, "auth_output": 1.5},
        metrics={
            "country_output": ["accuracy"],
            "auth_output": ["accuracy"],
        },
    )

    print("\n[*] Model Architecture:")
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════════
#  BUILD SKLEARN FALLBACK (if TF not available)
# ═══════════════════════════════════════════════════════════════════
def build_sklearn_model(X_train, y_country_train, y_auth_train):
    """Fallback: train sklearn Random Forest on flattened features."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("\n[*] Building sklearn fallback model...")

    # Flatten images to feature vectors
    X_flat = X_train.reshape(X_train.shape[0], -1)

    # Reduce dimensionality with PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, random_state=42)
    X_reduced = pca.fit_transform(X_flat)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # ── Country classifier ───────────────────────────────────────
    print("  Training country classifier...")
    country_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    country_clf.fit(X_scaled, y_country_train)
    country_acc = accuracy_score(y_country_train, country_clf.predict(X_scaled))
    print(f"    Country train accuracy: {country_acc*100:.1f}%")

    # ── Authenticity classifier ──────────────────────────────────
    print("  Training authenticity classifier...")
    auth_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=42
    )
    auth_clf.fit(X_scaled, y_auth_train)
    auth_acc = accuracy_score(y_auth_train, auth_clf.predict(X_scaled))
    print(f"    Authenticity train accuracy: {auth_acc*100:.1f}%")

    return {
        "country_clf": country_clf,
        "auth_clf": auth_clf,
        "scaler": scaler,
        "pca": pca,
        "currency_codes": CURRENCY_CODES,
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("  CURRENCY CNN TRAINING PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    X, y_country, y_auth = load_dataset()

    if len(X) == 0:
        print("\n[X] No images found! Run generate_dataset.py first.")
        return

    # ── Split train/test ─────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, yc_train, yc_test, ya_train, ya_test = train_test_split(
        X, y_country, y_auth, test_size=0.2, random_state=42, stratify=y_auth
    )

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Augment training data ────────────────────────────────────
    print("\n[*] Augmenting training data...")
    X_train_aug, yc_train_aug, ya_train_aug = augment_batch(
        X_train, yc_train, ya_train
    )
    print(f"  Augmented: {len(X_train)} -> {len(X_train_aug)} images")

    # ── Try TensorFlow first, fall back to sklearn ───────────────
    keras_success = False
    try:
        import tensorflow as tf
        print(f"\n[OK] TensorFlow {tf.__version__} found")

        model = build_keras_model(num_countries=len(CURRENCY_CODES))

        # ── Callbacks ────────────────────────────────────────────
        from tensorflow.keras.callbacks import (
            EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        )

        h5_path = os.path.join(MODEL_DIR, "currency_detector.h5")

        callbacks = [
            EarlyStopping(monitor="val_auth_output_accuracy",
                          patience=8, restore_best_weights=True),
            ModelCheckpoint(h5_path, monitor="val_auth_output_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=4, min_lr=1e-6),
        ]

        # ── Train ────────────────────────────────────────────────
        print("\n[*] Training CNN model...")
        history = model.fit(
            X_train_aug,
            {"country_output": yc_train_aug, "auth_output": ya_train_aug},
            validation_data=(
                X_test,
                {"country_output": yc_test, "auth_output": ya_test}
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Evaluate ─────────────────────────────────────────────
        print("\n[*] Evaluating on test set...")
        results = model.evaluate(
            X_test,
            {"country_output": yc_test, "auth_output": ya_test},
            verbose=0,
        )
        print(f"  Test Loss: {results[0]:.4f}")
        print(f"  Country Accuracy: {results[3]*100:.1f}%")
        print(f"  Authenticity Accuracy: {results[4]*100:.1f}%")

        # ── Save final model ─────────────────────────────────────
        model.save(h5_path)
        print(f"\n[SAVED] Keras model saved: {h5_path}")

        # Also save as SavedModel format
        saved_model_dir = os.path.join(MODEL_DIR, "currency_detector_saved")
        model.save(saved_model_dir)
        print(f"[SAVED] SavedModel saved: {saved_model_dir}")

        # Save training history
        hist_path = os.path.join(MODEL_DIR, "training_history.json")
        hist_data = {k: [float(v) for v in vals]
                     for k, vals in history.history.items()}
        with open(hist_path, "w") as f:
            json.dump(hist_data, f, indent=2)
        print(f"[SAVED] Training history: {hist_path}")

        keras_success = True

    except Exception as e:
        print(f"\n[!] TensorFlow training failed: {e}")
        print("  Falling back to sklearn model...")

    # ── Always also train sklearn backup ─────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING SKLEARN BACKUP MODEL")
    print("=" * 60)

    sklearn_model = build_sklearn_model(X_train_aug, yc_train_aug, ya_train_aug)

    # Evaluate sklearn on test set
    from sklearn.metrics import accuracy_score, classification_report
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_reduced = sklearn_model["pca"].transform(X_test_flat)
    X_test_scaled = sklearn_model["scaler"].transform(X_test_reduced)

    country_pred = sklearn_model["country_clf"].predict(X_test_scaled)
    auth_pred = sklearn_model["auth_clf"].predict(X_test_scaled)

    print(f"\n[*] Sklearn Test Results:")
    print(f"  Country Accuracy:       {accuracy_score(yc_test, country_pred)*100:.1f}%")
    print(f"  Authenticity Accuracy:  {accuracy_score(ya_test, auth_pred)*100:.1f}%")

    print(f"\n  Authenticity Report:")
    print(classification_report(ya_test, auth_pred,
                                 target_names=["REAL", "FAKE"]))

    # Save sklearn model
    pkl_path = os.path.join(MODEL_DIR, "currency_detector.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(sklearn_model, f)
    print(f"[SAVED] Sklearn model saved: {pkl_path}")

    # Save label mapping
    label_map = {i: code for i, code in enumerate(CURRENCY_CODES)}
    map_path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[SAVED] Label mapping saved: {map_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Models saved in: {MODEL_DIR}")
    if keras_success:
        print(f"     - currency_detector.h5     (Keras CNN)")
    print(f"     - currency_detector.pkl    (Sklearn backup)")
    print(f"     - label_mapping.json       (class labels)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    train()
