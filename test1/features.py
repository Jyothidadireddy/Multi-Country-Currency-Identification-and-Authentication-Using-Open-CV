"""
Feature Extractor — Computes image-based features for the ML model.
Extracts variance, skewness, curtosis, entropy from the image
using wavelet-like decomposition (same features as UCI dataset).
"""

import numpy as np
import cv2
from scipy import stats as scipy_stats


def extract_banknote_features(image):
    """
    Extract the 4 features used by the UCI Banknote Authentication dataset:
    variance, skewness, curtosis, entropy
    Computed from wavelet transform of the grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        BGR or grayscale image (camera frame or cropped note)

    Returns
    -------
    numpy.ndarray of shape (4,)
        [variance, skewness, curtosis, entropy]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to a consistent size for feature extraction
    gray = cv2.resize(gray, (400, 400))

    # ── Wavelet-like decomposition ───────────────────────────────
    # Apply Haar-like horizontal and vertical filters (approximation of DWT)
    # This mimics the wavelet transform used in the UCI dataset
    gray_f = gray.astype(np.float64)

    # Horizontal detail (high-frequency horizontal edges)
    h_detail = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    # Vertical detail (high-frequency vertical edges)
    v_detail = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)

    # Combined wavelet coefficients
    coeffs = np.concatenate([h_detail.ravel(), v_detail.ravel()])

    # ── Compute the 4 features ───────────────────────────────────
    variance = np.var(coeffs)
    skewness = float(scipy_stats.skew(coeffs))
    curtosis = float(scipy_stats.kurtosis(coeffs))
    entropy  = float(scipy_stats.entropy(np.histogram(coeffs, bins=256)[0] + 1e-10))

    # Normalise to match UCI dataset range (approximately)
    variance = np.clip(variance / 1000.0, -7, 7)
    skewness = np.clip(skewness, -14, 14)
    curtosis = np.clip(curtosis, -5, 18)
    entropy  = np.clip(entropy, -8, 3)

    return np.array([variance, skewness, curtosis, entropy])


def extract_image_quality_features(image):
    """
    Extract additional image quality features for better fake detection.

    Returns dict with:
    - sharpness: Laplacian variance (blurry = low)
    - edge_density: percentage of edge pixels
    - texture_complexity: number of unique intensity levels
    - color_uniformity: std of color channels
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, (400, 400))

    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Texture complexity
    texture_complexity = len(np.unique(gray)) / 256.0

    # Color uniformity (if colour image)
    if len(image.shape) == 3:
        color_uniformity = np.mean([np.std(image[:, :, c]) for c in range(3)])
    else:
        color_uniformity = np.std(gray)

    return {
        "sharpness": round(float(sharpness), 2),
        "edge_density": round(float(edge_density), 4),
        "texture_complexity": round(float(texture_complexity), 4),
        "color_uniformity": round(float(color_uniformity), 2),
    }
