"""
╔══════════════════════════════════════════════════════════════════╗
║  Synthetic Currency Dataset Generator                          ║
║  Generates REAL and FAKE note images for 17 countries          ║
║  20 real + 20 fake per country = 680 total images              ║
╚══════════════════════════════════════════════════════════════════╝

Each image is a 224×224 synthetic banknote-style training sample
with country-specific colours, text patterns, and visual features.

REAL notes: sharp text, proper colours, microprint, serial numbers,
            watermark patterns, fine guilloche patterns.
FAKE notes: blurry, colour-shifted, missing features, noise,
            poor alignment, missing security elements.
"""

import cv2
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Output settings ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
IMAGES_PER_CLASS = 20  # 20 real + 20 fake per currency
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dataset", "currency_images")

# ── Currency definitions ─────────────────────────────────────────
CURRENCIES = {
    "INR": {
        "name": "RESERVE BANK OF INDIA",
        "denom": "500",
        "sub_text": ["BHARATIYA", "MAHATMA GANDHI", "RBI", "GOVERNOR"],
        "primary_color": (180, 100, 50),     # saffron/orange tones
        "secondary_color": (220, 200, 170),
        "accent_color": (80, 60, 40),
    },
    "USD": {
        "name": "FEDERAL RESERVE NOTE",
        "denom": "20",
        "sub_text": ["UNITED STATES OF AMERICA", "IN GOD WE TRUST", "TREASURY", "DOLLAR"],
        "primary_color": (80, 120, 80),       # green
        "secondary_color": (200, 210, 200),
        "accent_color": (30, 60, 30),
    },
    "EUR": {
        "name": "EUROPEAN CENTRAL BANK",
        "denom": "50",
        "sub_text": ["EURO", "BCE", "ECB", "EUROSYSTEM"],
        "primary_color": (200, 140, 60),      # orange/yellow
        "secondary_color": (240, 230, 210),
        "accent_color": (100, 70, 30),
    },
    "GBP": {
        "name": "BANK OF ENGLAND",
        "denom": "20",
        "sub_text": ["POUND", "STERLING", "ELIZABETH", "PROMISE TO PAY"],
        "primary_color": (130, 80, 160),      # purple
        "secondary_color": (220, 210, 230),
        "accent_color": (60, 30, 80),
    },
    "NPR": {
        "name": "NEPAL RASTRA BANK",
        "denom": "100",
        "sub_text": ["NEPAL", "NEPALESE", "GOVERNOR", "RASTRA BANK"],
        "primary_color": (60, 130, 160),      # teal
        "secondary_color": (200, 220, 230),
        "accent_color": (30, 70, 90),
    },
    "JPY": {
        "name": "NIPPON GINKO",
        "denom": "10000",
        "sub_text": ["JAPAN", "YEN", "NIPPON GINKO"],
        "primary_color": (140, 120, 100),     # brown/beige
        "secondary_color": (230, 220, 210),
        "accent_color": (70, 50, 40),
    },
    "CNY": {
        "name": "PEOPLES BANK OF CHINA",
        "denom": "100",
        "sub_text": ["YUAN", "ZHONGGUO", "PEOPLES BANK"],
        "primary_color": (180, 50, 50),       # red
        "secondary_color": (230, 200, 200),
        "accent_color": (100, 20, 20),
    },
    "CAD": {
        "name": "BANK OF CANADA",
        "denom": "20",
        "sub_text": ["CANADA", "BANQUE DU CANADA", "DOLLAR"],
        "primary_color": (60, 140, 60),       # green
        "secondary_color": (200, 230, 200),
        "accent_color": (30, 80, 30),
    },
    "AUD": {
        "name": "RESERVE BANK OF AUSTRALIA",
        "denom": "50",
        "sub_text": ["AUSTRALIA", "COMMONWEALTH OF AUSTRALIA", "DOLLAR"],
        "primary_color": (200, 180, 50),      # gold/yellow
        "secondary_color": (240, 235, 200),
        "accent_color": (120, 100, 20),
    },
    "RUB": {
        "name": "BANK OF RUSSIA",
        "denom": "1000",
        "sub_text": ["RUBLE", "BANK OF RUSSIA"],
        "primary_color": (80, 120, 180),      # blue
        "secondary_color": (200, 210, 230),
        "accent_color": (30, 50, 100),
    },
    "BRL": {
        "name": "BANCO CENTRAL DO BRASIL",
        "denom": "50",
        "sub_text": ["REAL", "REAIS", "REPUBLICA FEDERATIVA"],
        "primary_color": (50, 130, 130),      # teal/green
        "secondary_color": (200, 230, 230),
        "accent_color": (20, 70, 70),
    },
    "KRW": {
        "name": "BANK OF KOREA",
        "denom": "10000",
        "sub_text": ["WON", "BANK OF KOREA"],
        "primary_color": (60, 160, 80),       # green
        "secondary_color": (200, 235, 210),
        "accent_color": (20, 90, 40),
    },
    "SAR": {
        "name": "SAUDI ARABIAN MONETARY",
        "denom": "100",
        "sub_text": ["RIYAL", "SAUDI ARABIAN MONETARY"],
        "primary_color": (100, 80, 140),      # purple/violet
        "secondary_color": (220, 210, 235),
        "accent_color": (50, 30, 80),
    },
    "AED": {
        "name": "CENTRAL BANK OF THE UAE",
        "denom": "100",
        "sub_text": ["DIRHAM", "UNITED ARAB EMIRATES"],
        "primary_color": (170, 130, 70),      # gold/brown
        "secondary_color": (235, 225, 200),
        "accent_color": (90, 60, 30),
    },
    "THB": {
        "name": "BANK OF THAILAND",
        "denom": "100",
        "sub_text": ["BAHT", "THAI", "GOVERNMENT OF THAILAND"],
        "primary_color": (180, 60, 60),       # red
        "secondary_color": (235, 210, 210),
        "accent_color": (100, 25, 25),
    },
    "LKR": {
        "name": "CENTRAL BANK OF SRI LANKA",
        "denom": "500",
        "sub_text": ["SRI LANKA", "CEYLON", "LANKA", "GOVERNOR"],
        "primary_color": (160, 130, 60),      # golden
        "secondary_color": (235, 225, 195),
        "accent_color": (90, 70, 20),
    },
    "BDT": {
        "name": "BANGLADESH BANK",
        "denom": "500",
        "sub_text": ["TAKA", "BANGLADESH", "PEOPLES REPUBLIC"],
        "primary_color": (80, 140, 80),       # green
        "secondary_color": (200, 230, 200),
        "accent_color": (30, 80, 30),
    },
}


# ═══════════════════════════════════════════════════════════════════
#  HELPER: Guilloche / fine-line pattern (simulates security print)
# ═══════════════════════════════════════════════════════════════════
def draw_guilloche(img, color, density=20, amplitude=10, thickness=1):
    """Draw wavy sine-pattern lines across the image (security engraving)."""
    h, w = img.shape[:2]
    for i in range(0, h, density):
        pts = []
        phase = random.uniform(0, 2 * np.pi)
        freq = random.uniform(0.02, 0.06)
        for x in range(0, w, 2):
            y = int(i + amplitude * np.sin(freq * x + phase))
            y = max(0, min(h - 1, y))
            pts.append((x, y))
        if len(pts) > 1:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_np], False, color, thickness, cv2.LINE_AA)
    return img


def draw_microprint(img, text, color, rows=6):
    """Draw tiny repeated text rows (simulates microprinting)."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(rows):
        y = int(h * 0.3 + i * 18)
        shift = random.randint(0, 40)
        line = (text + "  ") * 8
        cv2.putText(img, line, (shift, y), font, 0.45, color, 1, cv2.LINE_AA)
    return img


def draw_serial_number(img, color):
    """Draw a random serial/series number on the note."""
    serial = "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(10)])
    serial = serial[:3] + " " + serial[3:7] + " " + serial[7:]
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, serial, (int(w * 0.15), int(h * 0.92)), font, 0.5, color, 1, cv2.LINE_AA)
    return img


def draw_watermark_circle(img, center, radius, color, alpha=0.12):
    """Draw semi-transparent circular watermark."""
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Faint concentric rings
    for r in range(radius // 4, radius, radius // 4):
        cv2.circle(img, center, r, color, 1, cv2.LINE_AA)
    return img


# ═══════════════════════════════════════════════════════════════════
#  GENERATE REAL NOTE
# ═══════════════════════════════════════════════════════════════════
def generate_real_note(currency_code, variation_idx):
    """Generate a synthetic REAL banknote training image."""
    info = CURRENCIES[currency_code]
    w, h = IMG_SIZE

    # ── Base gradient background ─────────────────────────────────
    pc = info["primary_color"]
    sc = info["secondary_color"]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        for c in range(3):
            img[y, :, c] = int(sc[c] * (1 - ratio) + pc[c] * ratio)

    # Add subtle noise (paper texture)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Guilloche security pattern ───────────────────────────────
    gc = tuple(max(0, c - 30) for c in pc)
    dense = random.randint(15, 25)
    img = draw_guilloche(img, gc, density=dense, amplitude=random.randint(6, 14))

    # ── Microprint ───────────────────────────────────────────────
    micro_text = info["name"]
    mc = info["accent_color"]
    img = draw_microprint(img, micro_text, mc, rows=random.randint(4, 7))

    # ── Main bank name ───────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    ac = info["accent_color"]

    # Slight positional variation
    x_off = random.randint(-5, 5)
    y_off = random.randint(-3, 3)

    cv2.putText(img, info["name"], (10 + x_off, 25 + y_off),
                font, 0.42, ac, 1, cv2.LINE_AA)

    # ── Denomination (large) ─────────────────────────────────────
    denom = info["denom"]
    dx = random.randint(10, 30)
    dy = random.randint(60, 80)
    cv2.putText(img, denom, (dx, dy), font, 1.5, ac, 3, cv2.LINE_AA)

    # ── Sub-texts (security words) ───────────────────────────────
    sub = info["sub_text"]
    random.shuffle(sub)
    for i, txt in enumerate(sub[:3]):
        sx = random.randint(10, 50)
        sy = 105 + i * 22 + random.randint(-2, 2)
        cv2.putText(img, txt, (sx, sy), font, 0.38, ac, 1, cv2.LINE_AA)

    # ── Watermark ────────────────────────────────────────────────
    cx = random.randint(int(w * 0.6), int(w * 0.8))
    cy = random.randint(int(h * 0.3), int(h * 0.5))
    radius = random.randint(25, 40)
    wc = tuple(min(255, c + 60) for c in pc)
    img = draw_watermark_circle(img, (cx, cy), radius, wc)

    # ── Serial number ────────────────────────────────────────────
    img = draw_serial_number(img, ac)

    # ── Border (crisp, clean) ────────────────────────────────────
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), ac, 2)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), gc, 1)

    # ── Corner denomination marks ────────────────────────────────
    corners = [(w - 45, 25), (w - 45, h - 15)]
    for cx_c, cy_c in corners:
        cv2.putText(img, denom, (cx_c, cy_c), font, 0.5, ac, 1, cv2.LINE_AA)

    # ── Slight random rotation (real notes seen at slight angles)
    angle = random.uniform(-2, 2)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ── Brightness/contrast variation ────────────────────────────
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-8, 8)
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    return img


# ═══════════════════════════════════════════════════════════════════
#  GENERATE FAKE NOTE
# ═══════════════════════════════════════════════════════════════════
def generate_fake_note(currency_code, variation_idx):
    """Generate a synthetic FAKE banknote training image."""
    info = CURRENCIES[currency_code]
    w, h = IMG_SIZE

    # ── Deliberately wrong / shifted colours ─────────────────────
    pc = info["primary_color"]
    sc = info["secondary_color"]

    # Colour shift (fakes often have wrong tones)
    shift = random.randint(20, 60)
    direction = random.choice([-1, 1])
    pc_fake = tuple(max(0, min(255, c + direction * shift)) for c in pc)
    sc_fake = tuple(max(0, min(255, c + direction * shift // 2)) for c in sc)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        for c_idx in range(3):
            img[y, :, c_idx] = int(sc_fake[c_idx] * (1 - ratio) + pc_fake[c_idx] * ratio)

    # ── Heavy noise (poor print quality) ─────────────────────────
    noise = np.random.normal(0, random.randint(8, 20), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── NO guilloche OR very faint (fakes lack security engraving)
    if random.random() < 0.3:  # Only 30% chance of partial guilloche
        gc = tuple(max(0, c - 10) for c in pc_fake)
        img = draw_guilloche(img, gc, density=40, amplitude=5, thickness=1)

    # ── NO microprint (fakes miss this) ──────────────────────────
    # Skip microprint entirely

    # ── Bank name — may be WRONG, BLURRY, or MISSING ─────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    ac = info["accent_color"]
    ac_fake = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in ac)

    fake_type = random.choice(["wrong_text", "blurry", "missing", "misaligned"])

    if fake_type == "wrong_text":
        # Misspelled or wrong bank name
        wrong_names = [
            info["name"][::-1],  # reversed
            info["name"].replace("BANK", "BNAK"),
            info["name"].replace("RESERVE", "REESRVE"),
            "CENTRAL PRINTING PRESS",
            "NOVELTY NOTE",
        ]
        wrong = random.choice(wrong_names)
        cv2.putText(img, wrong, (10, 25), font, 0.38, ac_fake, 1, cv2.LINE_AA)
    elif fake_type == "blurry":
        # Draw text then blur the whole region
        cv2.putText(img, info["name"], (10, 25), font, 0.42, ac_fake, 1, cv2.LINE_AA)
        img[5:35, :] = cv2.GaussianBlur(img[5:35, :], (9, 9), 3)
    elif fake_type == "misaligned":
        # Text at wrong position/angle
        cv2.putText(img, info["name"], (30, 45), font, 0.35, ac_fake, 1, cv2.LINE_AA)
    # "missing" → no bank name at all

    # ── Denomination (may be wrong size or missing) ──────────────
    denom = info["denom"]
    if random.random() < 0.7:  # 70% have denomination
        dx = random.randint(5, 50)
        dy = random.randint(55, 90)
        thickness = random.choice([1, 2, 4])  # inconsistent
        cv2.putText(img, denom, (dx, dy), font, 1.3, ac_fake, thickness, cv2.LINE_AA)

    # ── Partial sub-text (missing most security words) ───────────
    sub = info["sub_text"]
    if random.random() < 0.4:  # only 40% have any sub text
        txt = random.choice(sub)
        cv2.putText(img, txt, (random.randint(10, 60), 120), font, 0.35, ac_fake, 1, cv2.LINE_AA)

    # ── NO watermark (key missing feature) ───────────────────────
    # Skip watermark entirely

    # ── NO serial number OR obviously fake one ───────────────────
    if random.random() < 0.3:
        fake_serial = "XXX 0000 000"
        cv2.putText(img, fake_serial, (int(w * 0.15), int(h * 0.92)),
                    font, 0.5, ac_fake, 1, cv2.LINE_AA)

    # ── Poor border (irregular, missing, or too thick) ───────────
    border_type = random.choice(["none", "thick", "partial"])
    if border_type == "thick":
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), ac_fake, 4)
    elif border_type == "partial":
        cv2.line(img, (0, 0), (w, 0), ac_fake, 2)
        cv2.line(img, (0, 0), (0, h), ac_fake, 2)
    # "none" → no border

    # ── Larger random rotation (poorly aligned scan/print) ───────
    angle = random.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ── Apply blur (fakes have poor print quality) ───────────────
    blur_ksize = random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), random.uniform(1, 3))

    # ── Brightness/contrast distortion ───────────────────────────
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-20, 20)
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    # ── Random JPEG artifacts (low quality) ──────────────────────
    if random.random() < 0.5:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(15, 40)]
        _, buf = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


# ═══════════════════════════════════════════════════════════════════
#  MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════
def generate_all():
    total = 0
    print("=" * 60)
    print("  SYNTHETIC CURRENCY DATASET GENERATOR")
    print(f"  {len(CURRENCIES)} countries x {IMAGES_PER_CLASS} real + {IMAGES_PER_CLASS} fake")
    print(f"  Total images: {len(CURRENCIES) * IMAGES_PER_CLASS * 2}")
    print("=" * 60)

    for code, info in CURRENCIES.items():
        real_dir = os.path.join(BASE_DIR, code, "real")
        fake_dir = os.path.join(BASE_DIR, code, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        # Generate REAL notes
        for i in range(IMAGES_PER_CLASS):
            img = generate_real_note(code, i)
            path = os.path.join(real_dir, f"{code}_real_{i+1:03d}.png")
            cv2.imwrite(path, img)
            total += 1

        # Generate FAKE notes
        for i in range(IMAGES_PER_CLASS):
            img = generate_fake_note(code, i)
            path = os.path.join(fake_dir, f"{code}_fake_{i+1:03d}.png")
            cv2.imwrite(path, img)
            total += 1

        print(f"  [OK] {code:4s} ({info['name'][:30]:30s}) -- {IMAGES_PER_CLASS} real + {IMAGES_PER_CLASS} fake")

    print("=" * 60)
    print(f"  DONE! Generated {total} images total")
    print(f"  Saved to: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all()
