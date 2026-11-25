"""
Configuration — Multi-Country Currency Voice Assistant
Supports 10+ countries with OCR-based rule detection
"""

# ── Supported Currencies (keyword → currency info) ─────────────
CURRENCY_RULES = {
    # ─── INDIA ───────────────────────────────────────────────────
    "INR": {
        "name": "Indian Rupee",
        "country": "India",
        "flag": "🇮🇳",
        "symbol": "₹",
        "keywords": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                      "RBI", "PAYABLE TO THE BEARER", "INDIA"],
        "denominations": [10, 20, 50, 100, 200, 500, 2000],
        "security_keywords": ["RESERVE BANK OF INDIA", "GOVERNOR", "GUARANTEE"],
        "weight": 3,  # high-confidence weight when keywords match
    },
    # ─── USA ─────────────────────────────────────────────────────
    "USD": {
        "name": "US Dollar",
        "country": "United States",
        "flag": "🇺🇸",
        "symbol": "$",
        "keywords": ["FEDERAL RESERVE", "UNITED STATES OF AMERICA", "THE UNITED STATES",
                      "IN GOD WE TRUST", "FEDERAL RESERVE NOTE", "TREASURY",
                      "DOLLAR", "WASHINGTON", "THIS NOTE IS LEGAL TENDER"],
        "denominations": [1, 2, 5, 10, 20, 50, 100],
        "security_keywords": ["FEDERAL RESERVE", "SECRETARY", "TREASURER", "TREASURY"],
        "weight": 3,
    },
    # ─── EURO ────────────────────────────────────────────────────
    "EUR": {
        "name": "Euro",
        "country": "European Union",
        "flag": "🇪🇺",
        "symbol": "€",
        "keywords": ["EURO", "EUROPEAN CENTRAL BANK", "BCE", "ECB", "EUROSYSTEM"],
        "denominations": [5, 10, 20, 50, 100, 200, 500],
        "security_keywords": ["CENTRAL BANK", "BCE", "ECB"],
        "weight": 3,
    },
    # ─── UK ──────────────────────────────────────────────────────
    "GBP": {
        "name": "British Pound",
        "country": "United Kingdom",
        "flag": "🇬🇧",
        "symbol": "£",
        "keywords": ["BANK OF ENGLAND", "POUND", "STERLING", "ELIZABETH", "CHARLES"],
        "denominations": [5, 10, 20, 50],
        "security_keywords": ["BANK OF ENGLAND", "PROMISE TO PAY"],
        "weight": 3,
    },
    # ─── JAPAN ───────────────────────────────────────────────────
    "JPY": {
        "name": "Japanese Yen",
        "country": "Japan",
        "flag": "🇯🇵",
        "symbol": "¥",
        "keywords": ["NIPPON GINKO", "JAPAN", "YEN", "日本銀行"],
        "denominations": [1000, 5000, 10000],
        "security_keywords": ["NIPPON GINKO", "JAPAN"],
        "weight": 2,
    },
    # ─── CHINA ───────────────────────────────────────────────────
    "CNY": {
        "name": "Chinese Yuan",
        "country": "China",
        "flag": "🇨🇳",
        "symbol": "¥",
        "keywords": ["PEOPLES BANK OF CHINA", "中国人民银行", "ZHONGGUO", "YUAN"],
        "denominations": [1, 5, 10, 20, 50, 100],
        "security_keywords": ["PEOPLES BANK", "中国人民银行"],
        "weight": 2,
    },
    # ─── CANADA ──────────────────────────────────────────────────
    "CAD": {
        "name": "Canadian Dollar",
        "country": "Canada",
        "flag": "🇨🇦",
        "symbol": "C$",
        "keywords": ["BANK OF CANADA", "BANQUE DU CANADA", "CANADA"],
        "denominations": [5, 10, 20, 50, 100],
        "security_keywords": ["BANK OF CANADA", "BANQUE DU CANADA"],
        "weight": 3,
    },
    # ─── AUSTRALIA ───────────────────────────────────────────────
    "AUD": {
        "name": "Australian Dollar",
        "country": "Australia",
        "flag": "🇦🇺",
        "symbol": "A$",
        "keywords": ["RESERVE BANK OF AUSTRALIA", "AUSTRALIA", "COMMONWEALTH OF AUSTRALIA"],
        "denominations": [5, 10, 20, 50, 100],
        "security_keywords": ["RESERVE BANK OF AUSTRALIA"],
        "weight": 3,
    },
    # ─── RUSSIA ──────────────────────────────────────────────────
    "RUB": {
        "name": "Russian Ruble",
        "country": "Russia",
        "flag": "🇷🇺",
        "symbol": "₽",
        "keywords": ["BANK OF RUSSIA", "БАНК РОССИИ", "РУБЛЕЙ", "RUBLE"],
        "denominations": [50, 100, 200, 500, 1000, 2000, 5000],
        "security_keywords": ["BANK OF RUSSIA", "БАНК РОССИИ"],
        "weight": 2,
    },
    # ─── BRAZIL ──────────────────────────────────────────────────
    "BRL": {
        "name": "Brazilian Real",
        "country": "Brazil",
        "flag": "🇧🇷",
        "symbol": "R$",
        "keywords": ["BANCO CENTRAL DO BRASIL", "REPUBLICA FEDERATIVA", "REAL", "REAIS"],
        "denominations": [2, 5, 10, 20, 50, 100, 200],
        "security_keywords": ["BANCO CENTRAL DO BRASIL"],
        "weight": 3,
    },
    # ─── SOUTH KOREA ─────────────────────────────────────────────
    "KRW": {
        "name": "South Korean Won",
        "country": "South Korea",
        "flag": "🇰🇷",
        "symbol": "₩",
        "keywords": ["BANK OF KOREA", "한국은행", "WON"],
        "denominations": [1000, 5000, 10000, 50000],
        "security_keywords": ["BANK OF KOREA", "한국은행"],
        "weight": 2,
    },
    # ─── SAUDI ARABIA ────────────────────────────────────────────
    "SAR": {
        "name": "Saudi Riyal",
        "country": "Saudi Arabia",
        "flag": "🇸🇦",
        "symbol": "﷼",
        "keywords": ["SAUDI ARABIAN MONETARY", "RIYAL", "المملكة العربية السعودية"],
        "denominations": [1, 5, 10, 50, 100, 500],
        "security_keywords": ["SAUDI ARABIAN MONETARY"],
        "weight": 2,
    },
    # ─── UAE ─────────────────────────────────────────────────────
    "AED": {
        "name": "UAE Dirham",
        "country": "United Arab Emirates",
        "flag": "🇦🇪",
        "symbol": "د.إ",
        "keywords": ["CENTRAL BANK OF THE UAE", "UNITED ARAB EMIRATES", "DIRHAM"],
        "denominations": [5, 10, 20, 50, 100, 200, 500, 1000],
        "security_keywords": ["CENTRAL BANK OF THE UAE"],
        "weight": 2,
    },
    # ─── NEPAL ───────────────────────────────────────────────────
    "NPR": {
        "name": "Nepalese Rupee",
        "country": "Nepal",
        "flag": "🇳🇵",
        "symbol": "रू",
        "keywords": ["NEPAL RASTRA BANK", "NEPAL", "नेपाल राष्ट्र बैंक",
                      "RASTRA BANK", "NEPALESE", "NEPAL RASTRA"],
        "denominations": [5, 10, 20, 50, 100, 500, 1000],
        "security_keywords": ["NEPAL RASTRA BANK", "GOVERNOR", "नेपाल राष्ट्र बैंक"],
        "weight": 3,
    },
    # ─── THAILAND ────────────────────────────────────────────────
    "THB": {
        "name": "Thai Baht",
        "country": "Thailand",
        "flag": "🇹🇭",
        "symbol": "฿",
        "keywords": ["BANK OF THAILAND", "THAILAND", "BAHT", "ธนาคารแห่งประเทศไทย",
                      "THAI", "GOVERNMENT OF THAILAND"],
        "denominations": [20, 50, 100, 500, 1000],
        "security_keywords": ["BANK OF THAILAND", "ธนาคารแห่งประเทศไทย"],
        "weight": 2,
    },
    # ─── SRI LANKA ──────────────────────────────────────────────
    "LKR": {
        "name": "Sri Lankan Rupee",
        "country": "Sri Lanka",
        "flag": "🇱🇰",
        "symbol": "Rs",
        "keywords": ["CENTRAL BANK OF SRI LANKA", "SRI LANKA",
                      "ශ්\u200dරී ලංකා", "CEYLON", "LANKA"],
        "denominations": [20, 50, 100, 500, 1000, 2000, 5000],
        "security_keywords": ["CENTRAL BANK OF SRI LANKA", "GOVERNOR"],
        "weight": 3,
    },
    # ─── BANGLADESH ─────────────────────────────────────────────
    "BDT": {
        "name": "Bangladeshi Taka",
        "country": "Bangladesh",
        "flag": "🇧🇩",
        "symbol": "৳",
        "keywords": ["BANGLADESH BANK", "TAKA", "BANGLADESH", "বাংলাদেশ ব্যাংক",
                      "PEOPLES REPUBLIC OF BANGLADESH"],
        "denominations": [2, 5, 10, 20, 50, 100, 500, 1000],
        "security_keywords": ["BANGLADESH BANK", "GOVERNOR", "বাংলাদেশ ব্যাংক"],
        "weight": 3,
    },
}

# ── Anti-Integrity Scoring ───────────────────────────────────────
SCORE_THRESHOLDS = {
    "REAL":       3,   # score >= 3 → likely genuine (lowered for better accuracy)
    "SUSPICIOUS": 1.5, # score >= 1.5 → suspicious
    # score < 1.5 → FAKE
}

# ── OCR Confidence Threshold ─────────────────────────────────────
OCR_CONFIDENCE = 0.20   # lowered to catch more text even with poor camera

# ── Detection Cooldown (seconds between voice announcements) ─────
VOICE_COOLDOWN = 3.0

# ── Camera Settings ──────────────────────────────────────────────
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_INDEX = 0

# ── ML Model Settings ───────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ML_MODEL_PATH = os.path.join(MODEL_DIR, "banknote_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
ML_CONFIDENCE_WEIGHT = 0.4   # how much the ML model influences the final score

# ── Image-Based Model Settings (US/UK upload detection) ──────────
IMAGE_COUNTRY_MODEL_PATH = os.path.join(MODEL_DIR, "image_country_clf.pkl")
IMAGE_AUTH_MODEL_PATH = os.path.join(MODEL_DIR, "image_auth_clf.pkl")
IMAGE_FEATURE_PIPELINE_PATH = os.path.join(MODEL_DIR, "image_feature_pipeline.pkl")
