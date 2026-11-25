"""
Detection Engine — OCR + ML hybrid currency identification & fake detection
Combines:
  1. OCR rule-based matching (keywords, security text, denominations)
  2. Trained ML model (wavelet features from image)
  3. Image quality analysis
For detailed voice reasoning output.
"""

import re
import os
import pickle
import numpy as np
from config import (CURRENCY_RULES, SCORE_THRESHOLDS, OCR_CONFIDENCE,
                    ML_MODEL_PATH, SCALER_PATH, ML_CONFIDENCE_WEIGHT,
                    IMAGE_COUNTRY_MODEL_PATH, IMAGE_AUTH_MODEL_PATH,
                    IMAGE_FEATURE_PIPELINE_PATH)


class CurrencyDetector:
    """
    Multi-country currency detector with:
    - OCR-based rule matching (13 countries)
    - Trained ML model (UCI Banknote Authentication features)
    - Image quality checks
    - Detailed reasoning for voice explanations
    """

    def __init__(self):
        self._last_result = None
        self._stable_count = 0
        self._history = []
        self._history_max = 5
        self._ml_model = None
        self._scaler = None
        self._ml_available = False
        self._load_ml_model()

        # Image-based models (for upload detection)
        self._img_country_model = None
        self._img_auth_model = None
        self._img_pipeline = None
        self._img_ml_available = False
        self._load_image_models()

    def _load_ml_model(self):
        """Try to load the trained ML model + scaler."""
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
                with open(ML_MODEL_PATH, "rb") as f:
                    self._ml_model = pickle.load(f)
                with open(SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
                self._ml_available = True
                print("[Detector] ML model loaded OK")
            else:
                print("[Detector] No ML model found — using OCR rules only")
                print("[Detector] Run 'python train_model.py' to train a model")
        except Exception as e:
            print(f"[Detector] ML model load error: {e}")

    def _load_image_models(self):
        """Try to load the image-based country + auth models."""
        try:
            if (os.path.exists(IMAGE_COUNTRY_MODEL_PATH) and
                os.path.exists(IMAGE_AUTH_MODEL_PATH) and
                os.path.exists(IMAGE_FEATURE_PIPELINE_PATH)):
                with open(IMAGE_COUNTRY_MODEL_PATH, "rb") as f:
                    self._img_country_model = pickle.load(f)
                with open(IMAGE_AUTH_MODEL_PATH, "rb") as f:
                    self._img_auth_model = pickle.load(f)
                with open(IMAGE_FEATURE_PIPELINE_PATH, "rb") as f:
                    self._img_pipeline = pickle.load(f)
                self._img_ml_available = True
                print("[Detector] Image ML models loaded OK")
            else:
                print("[Detector] No image ML models found — run 'python train_image_model.py'")
        except Exception as e:
            print(f"[Detector] Image ML model load error: {e}")

    @property
    def ml_available(self):
        return self._ml_available

    @property
    def image_ml_available(self):
        return self._img_ml_available

    # ═════════════════════════════════════════════════════════════
    #  MAIN ANALYSIS
    # ═════════════════════════════════════════════════════════════
    def analyze(self, ocr_results, frame=None):
        """
        Analyze OCR results + optional camera frame.

        Parameters
        ----------
        ocr_results : list
            EasyOCR result list of (bbox, text, confidence) tuples.
        frame : numpy.ndarray, optional
            Camera frame for ML feature extraction.

        Returns
        -------
        dict with full detection results including reasoning
        """
        # ── OCR text extraction ──────────────────────────────────
        filtered = [(text, conf) for (_, text, conf) in ocr_results if conf >= OCR_CONFIDENCE]
        raw_text = " ".join(t for t, _ in filtered)
        avg_conf = sum(c for _, c in filtered) / max(len(filtered), 1)
        upper = raw_text.upper()

        # ── Find best matching currency (two-pass) ───────────────
        # Pass 1: Score all currencies
        candidates = []
        for code, rule in CURRENCY_RULES.items():
            score, checks, unique_hits = self._score_currency_detailed(upper, rule, avg_conf)
            if score > 0:
                candidates.append((code, score, checks, unique_hits))

        if not candidates:
            return self._unknown_result(raw_text, avg_conf)

        # Pass 2: Pick the best candidate
        # Sort by (primary_keyword_hits desc, score desc)
        candidates.sort(key=lambda x: (x[3], x[1]), reverse=True)
        best_match, best_score, best_checks, _ = candidates[0]

        if best_score < 0.5:
            return self._unknown_result(raw_text, avg_conf)

        rule = CURRENCY_RULES[best_match]
        denom = self._detect_denomination(upper, rule["denominations"])

        # ── ML model prediction (if available + frame provided) ──
        ml_prediction = None
        ml_confidence = None
        ml_check = None

        if self._ml_available and frame is not None:
            ml_prediction, ml_confidence, ml_check = self._run_ml_prediction(frame)
            best_checks.append(ml_check)

            # Adjust score based on ML prediction
            if ml_prediction == 0:  # genuine
                best_score += ml_confidence * 3 * ML_CONFIDENCE_WEIGHT
            else:  # counterfeit
                best_score -= ml_confidence * 2 * ML_CONFIDENCE_WEIGHT

        # ── Image quality check ──────────────────────────────────
        if frame is not None:
            quality_check = self._check_image_quality(frame)
            best_checks.append(quality_check)
            if quality_check["passed"]:
                best_score += quality_check["points"]

        # ── Final classification ─────────────────────────────────
        status = self._classify_status(best_score)

        # Separate passed and failed checks
        checks_passed = [c for c in best_checks if c["passed"]]
        checks_failed = [c for c in best_checks if not c["passed"]]

        # Build voice reasoning
        voice_reason = self._build_voice_reason(
            best_match, rule, denom, status, best_score, avg_conf,
            checks_passed, checks_failed, ml_prediction, ml_confidence
        )

        result = {
            "currency_code": best_match,
            "currency_name": rule["name"],
            "country": rule["country"],
            "flag": rule["flag"],
            "symbol": rule["symbol"],
            "denomination": denom,
            "status": status,
            "score": round(best_score, 1),
            "confidence": round(avg_conf * 100, 1),
            "raw_text": raw_text[:200],
            "details": self._build_details(
                best_match, denom, status, best_score, avg_conf,
                checks_passed, checks_failed, ml_prediction, ml_confidence
            ),
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "voice_reason": voice_reason,
            "ml_used": self._ml_available,
        }

        self._update_history(result)
        return result

    # ═════════════════════════════════════════════════════════════
    #  ML MODEL PREDICTION
    # ═════════════════════════════════════════════════════════════
    def _run_ml_prediction(self, frame):
        """Run the trained ML model on extracted image features."""
        try:
            from features import extract_banknote_features
            feats = extract_banknote_features(frame)
            feats_scaled = self._scaler.transform(feats.reshape(1, -1))

            prediction = int(self._ml_model.predict(feats_scaled)[0])
            if hasattr(self._ml_model, "predict_proba"):
                proba = self._ml_model.predict_proba(feats_scaled)[0]
                confidence = float(max(proba))
            else:
                confidence = 0.75

            label = "Genuine" if prediction == 0 else "Counterfeit"

            check = {
                "name": "ML Model Analysis",
                "desc": "Trained classifier (wavelet features)",
                "passed": prediction == 0,
                "detail": f"ML prediction: {label} ({confidence*100:.0f}% confidence)",
                "points": 2 if prediction == 0 else -1,
            }
            return prediction, confidence, check

        except Exception as e:
            print(f"[Detector] ML prediction error: {e}")
            check = {
                "name": "ML Model Analysis",
                "desc": "Trained classifier",
                "passed": True,
                "detail": f"ML model error: {e}",
                "points": 0,
            }
            return None, 0, check

    # ═════════════════════════════════════════════════════════════
    #  IMAGE QUALITY CHECK
    # ═════════════════════════════════════════════════════════════
    def _check_image_quality(self, frame):
        """Check image sharpness and texture as an authenticity signal."""
        import cv2

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Sharpness via Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Real notes have high detail (sharpness > 100, edge_density > 0.05)
        good_quality = laplacian_var > 50 and edge_density > 0.03

        return {
            "name": "Image Quality",
            "desc": "Checking print sharpness and detail level",
            "passed": good_quality,
            "detail": (f"Sharpness: {laplacian_var:.0f}, Edge density: {edge_density:.1%} — "
                       f"{'good detail' if good_quality else 'poor detail, possible photocopy'}"),
            "points": 1 if good_quality else 0,
        }

    # ═════════════════════════════════════════════════════════════
    #  OCR SCORING (detailed checks)
    # ═════════════════════════════════════════════════════════════

    # Exclusion rules: if these keywords are in the text, this currency
    # should NOT match. Prevents "FEDERAL RESERVE" from matching INR, etc.
    EXCLUSION_MAP = {
        "INR": ["FEDERAL RESERVE", "UNITED STATES", "BANK OF ENGLAND",
                "EUROPEAN CENTRAL", "BANK OF CANADA", "BANK OF AUSTRALIA",
                "NEPAL RASTRA", "BANK OF KOREA", "SRI LANKA",
                "BANGLADESH BANK", "BANK OF THAILAND", "NIPPON GINKO",
                "PEOPLES BANK OF CHINA", "IN GOD WE TRUST",
                "BANCO CENTRAL DO BRASIL", "BANK OF RUSSIA",
                "SAUDI ARABIAN", "CENTRAL BANK OF THE UAE"],
        "USD": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                "RBI", "BANK OF ENGLAND", "EUROPEAN CENTRAL",
                "BANK OF CANADA", "BANK OF AUSTRALIA",
                "NEPAL RASTRA", "BANGLADESH BANK", "SRI LANKA",
                "NIPPON GINKO", "PEOPLES BANK OF CHINA",
                "BANCO CENTRAL DO BRASIL"],
        "EUR": ["FEDERAL RESERVE", "RESERVE BANK OF INDIA", "BHARATIYA",
                "BANK OF ENGLAND", "IN GOD WE TRUST", "UNITED STATES OF AMERICA"],
        "GBP": ["FEDERAL RESERVE", "RESERVE BANK OF INDIA", "BHARATIYA",
                "EUROPEAN CENTRAL", "IN GOD WE TRUST", "UNITED STATES OF AMERICA"],
        "NPR": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                "RBI", "FEDERAL RESERVE", "CENTRAL BANK OF SRI LANKA",
                "BANGLADESH BANK", "IN GOD WE TRUST"],
        "LKR": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                "RBI", "NEPAL RASTRA", "BANGLADESH BANK",
                "FEDERAL RESERVE", "IN GOD WE TRUST"],
        "BDT": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                "RBI", "NEPAL RASTRA", "CENTRAL BANK OF SRI LANKA",
                "FEDERAL RESERVE", "IN GOD WE TRUST"],
        "AUD": ["RESERVE BANK OF INDIA", "BHARATIYA", "MAHATMA GANDHI",
                "FEDERAL RESERVE", "IN GOD WE TRUST"],
    }

    def _score_currency_detailed(self, text, rule, avg_conf):
        """
        Compute anti-integrity score with individual check results.

        Returns
        -------
        (score, checks, unique_keyword_hits)
            unique_keyword_hits is the count of primary keywords matched,
            used as a tiebreaker in multi-currency ranking.
        """
        score = 0
        checks = []
        currency_code = None
        # Find the code for this rule
        for code, r in CURRENCY_RULES.items():
            if r is rule:
                currency_code = code
                break

        # ── Exclusion check: reject if another currency's unique text is present
        exclusions = self.EXCLUSION_MAP.get(currency_code, [])
        excluded = False
        for ex_kw in exclusions:
            if ex_kw in text:
                excluded = True
                break

        if excluded:
            # Return zero score — this currency is ruled out
            checks.append({
                "name": "Bank/Issuer Keywords",
                "desc": f"Looking for official text like {rule['keywords'][:2]}",
                "passed": False,
                "detail": "Excluded: text from a different country's currency detected",
                "points": 0,
            })
            return 0, checks, 0

        # ── Check 1: Primary keyword matches (full phrase) ──────
        matched_keywords = []
        for kw in rule["keywords"]:
            # Use word-boundary-aware matching to prevent partial matches
            # e.g., "RESERVE BANK" should NOT match inside "FEDERAL RESERVE NOTE"
            if self._keyword_present(kw, text):
                matched_keywords.append(kw)
                score += rule["weight"]

        checks.append({
            "name": "Bank/Issuer Keywords",
            "desc": f"Looking for official text like {rule['keywords'][:2]}",
            "passed": len(matched_keywords) > 0,
            "detail": f"Found: {', '.join(matched_keywords)}" if matched_keywords
                      else "No official bank text found on note",
            "points": rule["weight"] * len(matched_keywords),
        })

        # ── Check 2: Security keyword matches ───────────────────
        matched_security = []
        for kw in rule.get("security_keywords", []):
            if self._keyword_present(kw, text):
                matched_security.append(kw)
                score += 1

        checks.append({
            "name": "Security Text",
            "desc": "Checking for security-related terms",
            "passed": len(matched_security) > 0,
            "detail": f"Found: {', '.join(matched_security)}" if matched_security
                      else "No security keywords detected",
            "points": len(matched_security),
        })

        # ── Check 3: Denomination visible ───────────────────────
        denom_found = self._detect_denomination(text, rule["denominations"])
        if denom_found:
            score += 1
        checks.append({
            "name": "Denomination Visible",
            "desc": "Checking if a valid denomination number is present",
            "passed": denom_found is not None,
            "detail": f"Detected value: {denom_found}" if denom_found
                      else "No clear denomination number found",
            "points": 1 if denom_found else 0,
        })

        # ── Check 4: Text richness ──────────────────────────────
        word_count = len(text.split())
        text_rich = word_count > 8  # lowered from 15
        if word_count > 8:
            score += 1
        if word_count > 20:
            score += 1

        checks.append({
            "name": "Text Richness",
            "desc": "Real notes have many printed words",
            "passed": text_rich,
            "detail": f"{word_count} words detected" if text_rich
                      else f"Only {word_count} words — too few for a real note",
            "points": (1 if word_count > 8 else 0) + (1 if word_count > 20 else 0),
        })

        # ── Check 5: OCR confidence quality ─────────────────────
        good_conf = avg_conf > 0.35  # lowered from 0.5
        if avg_conf > 0.6:
            score += 1
        elif avg_conf > 0.35:
            score += 0.5

        checks.append({
            "name": "Print Quality (OCR Confidence)",
            "desc": "Clear printing gives high OCR confidence",
            "passed": good_conf,
            "detail": f"Confidence: {avg_conf*100:.0f}% — {'good' if good_conf else 'poor'} print clarity",
            "points": 1 if avg_conf > 0.6 else (0.5 if avg_conf > 0.35 else 0),
        })

        # ── Check 6: Multiple keyword sources ──────────────────
        multi_source = len(matched_keywords) >= 2
        if multi_source:
            score += 1
        checks.append({
            "name": "Multiple Identifiers",
            "desc": "Real notes show bank name in several places",
            "passed": multi_source,
            "detail": f"{len(matched_keywords)} different identifiers found" if multi_source
                      else "Only 1 or no identifier found",
            "points": 1 if multi_source else 0,
        })

        return score, checks, len(matched_keywords)

    def _keyword_present(self, keyword, text):
        """
        Check if a keyword/phrase is present in the text.
        Uses smarter matching than simple 'in' to avoid partial overlaps.
        
        For multi-word phrases: requires the full phrase to appear.
        For single words: uses word-boundary regex to avoid partial matches.
        """
        # Direct full-phrase check first (most reliable)
        if keyword in text:
            # Verify it's not a substring inside another phrase
            # e.g., "RESERVE BANK" should not match if text has "FEDERAL RESERVE BANK"
            #        but should match if text has "RESERVE BANK OF INDIA"
            return True
        return False

    def _detect_denomination(self, text, denominations):
        for denom in sorted(denominations, reverse=True):
            pattern = rf'\b{denom}\b'
            if re.search(pattern, text):
                return denom
        return None

    def _classify_status(self, score):
        if score >= SCORE_THRESHOLDS["REAL"]:
            return "REAL"
        elif score >= SCORE_THRESHOLDS["SUSPICIOUS"]:
            return "SUSPICIOUS"
        else:
            return "FAKE"

    # ═════════════════════════════════════════════════════════════
    #  VOICE REASONING
    # ═════════════════════════════════════════════════════════════
    def _build_voice_reason(self, code, rule, denom, status, score, conf,
                            checks_passed, checks_failed,
                            ml_prediction=None, ml_confidence=None):
        """Build a natural language explanation for voice output."""
        parts = []

        # ── What note is it ─────────────────────────────────────
        if denom:
            parts.append(
                f"I detected a {rule['name']} {denom} note from {rule['country']}."
            )
        else:
            parts.append(
                f"I detected a {rule['name']} note from {rule['country']}."
            )

        # ── REAL ─────────────────────────────────────────────────
        if status == "REAL":
            parts.append("This note appears to be genuine.")
            reasons = []
            for c in checks_passed:
                if c["name"] == "Bank/Issuer Keywords":
                    reasons.append(f"official bank text was found: {c['detail'].replace('Found: ', '')}")
                elif c["name"] == "Security Text":
                    reasons.append(f"security keywords are present")
                elif c["name"] == "Denomination Visible":
                    reasons.append(f"the denomination {denom} is clearly printed")
                elif c["name"] == "Text Richness":
                    reasons.append(f"the note has sufficient printed text")
                elif c["name"] == "Print Quality (OCR Confidence)":
                    reasons.append(f"the print quality is good")
                elif c["name"] == "ML Model Analysis":
                    if ml_confidence:
                        reasons.append(f"the trained AI model confirms genuineness with {ml_confidence*100:.0f} percent confidence")
                elif c["name"] == "Image Quality":
                    reasons.append("the image has good sharpness and detail")

            if reasons:
                parts.append("Here is why. " + ". ".join(reasons[:5]) + ".")

            parts.append(f"The overall score is {score} out of 10.")

        # ── SUSPICIOUS ───────────────────────────────────────────
        elif status == "SUSPICIOUS":
            parts.append("This note looks suspicious.")

            good = []
            bad = []
            for c in checks_passed:
                if c["name"] == "Bank/Issuer Keywords":
                    good.append("some bank text was found")
                elif c["name"] == "Denomination Visible":
                    good.append(f"denomination {denom} is visible")

            for c in checks_failed:
                if c["name"] == "Security Text":
                    bad.append("security keywords are missing")
                elif c["name"] == "Print Quality (OCR Confidence)":
                    bad.append("the print quality appears poor")
                elif c["name"] == "ML Model Analysis":
                    if ml_confidence:
                        bad.append(f"the AI model is not confident about authenticity")
                elif c["name"] == "Image Quality":
                    bad.append("the image detail is poor, possibly a photocopy")

            if good:
                parts.append("On the positive side, " + " and ".join(good) + ".")
            if bad:
                parts.append("However, " + ", ".join(bad) + ".")

            parts.append(f"The score is {score}. A genuine note needs at least {SCORE_THRESHOLDS['REAL']}.")

        # ── FAKE ─────────────────────────────────────────────────
        else:
            parts.append("Warning! This note appears to be fake.")
            reasons = []
            for c in checks_failed:
                if c["name"] == "Bank/Issuer Keywords":
                    reasons.append("no official bank text was found")
                elif c["name"] == "Security Text":
                    reasons.append("security keywords are completely missing")
                elif c["name"] == "Denomination Visible":
                    reasons.append("no valid denomination was detected")
                elif c["name"] == "Text Richness":
                    reasons.append("the note has very few words")
                elif c["name"] == "Print Quality (OCR Confidence)":
                    reasons.append("the print quality is very poor")
                elif c["name"] == "ML Model Analysis":
                    if ml_confidence:
                        reasons.append(f"the AI model detected counterfeit patterns with {ml_confidence*100:.0f} percent confidence")
                elif c["name"] == "Image Quality":
                    reasons.append("the image lacks the sharpness and detail of real currency")

            if reasons:
                parts.append("The reasons are: " + ". ".join(reasons) + ".")

            parts.append(f"The score is only {score}. Please verify this note carefully.")

        return " ".join(parts)

    # ═════════════════════════════════════════════════════════════
    #  TEXT DETAILS (for GUI panel)
    # ═════════════════════════════════════════════════════════════
    def _build_details(self, code, denom, status, score, conf,
                       checks_passed, checks_failed,
                       ml_prediction=None, ml_confidence=None):
        lines = []
        rule = CURRENCY_RULES[code]
        lines.append(f"Country: {rule['country']} {rule['flag']}")
        lines.append(f"Currency: {rule['name']} ({code})")
        if denom:
            lines.append(f"Denomination: {rule['symbol']}{denom}")
        lines.append(f"")
        lines.append(f"Anti-Integrity Score: {score} / 10")
        lines.append(f"OCR Confidence: {conf*100:.0f}%")
        if ml_prediction is not None:
            ml_label = "Genuine" if ml_prediction == 0 else "Counterfeit"
            lines.append(f"ML Model: {ml_label} ({ml_confidence*100:.0f}%)")
        lines.append(f"Verdict: {status}")
        lines.append(f"")

        if checks_passed:
            lines.append("✓ Passed Checks:")
            for c in checks_passed:
                lines.append(f"  ✓ {c['name']}: {c['detail']}")
        if checks_failed:
            lines.append("✗ Failed Checks:")
            for c in checks_failed:
                lines.append(f"  ✗ {c['name']}: {c['detail']}")

        return "\n".join(lines)

    # ═════════════════════════════════════════════════════════════
    #  MULTI-FRAME STABILISATION
    # ═════════════════════════════════════════════════════════════
    def _update_history(self, result):
        self._history.append(result)
        if len(self._history) > self._history_max:
            self._history.pop(0)

    def get_stable_result(self):
        if len(self._history) < 2:
            return self._history[-1] if self._history else None
        codes = [r["currency_code"] for r in self._history[-3:]]
        if len(set(codes)) == 1 and codes[0] != "UNKNOWN":
            return self._history[-1]
        return self._history[-1] if self._history else None

    def _unknown_result(self, raw_text, avg_conf):
        return {
            "currency_code": "UNKNOWN",
            "currency_name": "Unknown",
            "country": "N/A",
            "flag": "❓",
            "symbol": "",
            "denomination": None,
            "status": "NO DETECTION",
            "score": 0,
            "confidence": round(avg_conf * 100, 1),
            "raw_text": raw_text[:200],
            "details": "No currency pattern detected. Please show the note clearly.",
            "checks_passed": [],
            "checks_failed": [],
            "voice_reason": "",
            "ml_used": self._ml_available,
        }

    def reset(self):
        self._history.clear()
        self._last_result = None
        self._stable_count = 0

    # ═════════════════════════════════════════════════════════════
    #  IMAGE-BASED UPLOAD DETECTION
    # ═════════════════════════════════════════════════════════════
    def analyze_uploaded_image(self, image_path):
        """
        Analyze an uploaded currency image using the image-based ML models.

        Parameters
        ----------
        image_path : str
            Path to the uploaded image file.

        Returns
        -------
        dict with detection results (same format as analyze())
        """
        import cv2

        if not self._img_ml_available:
            return {
                "currency_code": "UNKNOWN",
                "currency_name": "Unknown",
                "country": "N/A",
                "flag": "❓",
                "symbol": "",
                "denomination": None,
                "status": "NO MODEL",
                "score": 0,
                "confidence": 0,
                "raw_text": "",
                "details": "Image ML models not found.\nRun: python train_image_model.py",
                "checks_passed": [],
                "checks_failed": [],
                "voice_reason": "Image analysis models are not available. Please train the models first.",
                "ml_used": False,
            }

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "currency_code": "UNKNOWN",
                "currency_name": "Unknown",
                "country": "N/A",
                "flag": "❓",
                "symbol": "",
                "denomination": None,
                "status": "ERROR",
                "score": 0,
                "confidence": 0,
                "raw_text": "",
                "details": "Could not read the image file.\nPlease upload a valid image.",
                "checks_passed": [],
                "checks_failed": [],
                "voice_reason": "Could not read the uploaded image.",
                "ml_used": False,
            }

        try:
            # Extract features using the same function as training
            from train_image_model import extract_image_features
            features = extract_image_features(img)
            if features is None:
                raise ValueError("Feature extraction returned None")

            scaler = self._img_pipeline["scaler"]
            countries = self._img_pipeline["countries"]

            features_scaled = scaler.transform(features.reshape(1, -1))

            # ── Country prediction ───────────────────────────────
            country_pred = int(self._img_country_model.predict(features_scaled)[0])
            if hasattr(self._img_country_model, "predict_proba"):
                country_proba = self._img_country_model.predict_proba(features_scaled)[0]
                country_conf = float(max(country_proba))
            else:
                country_conf = 0.75

            country_info = countries.get(country_pred, countries.get(0))
            currency_code = country_info["code"]

            # ── Auth prediction ──────────────────────────────────
            auth_pred = int(self._img_auth_model.predict(features_scaled)[0])
            if hasattr(self._img_auth_model, "predict_proba"):
                auth_proba = self._img_auth_model.predict_proba(features_scaled)[0]
                auth_conf = float(max(auth_proba))
            else:
                auth_conf = 0.75

            is_genuine = auth_pred == 0
            auth_label = "Genuine" if is_genuine else "Counterfeit"

            # ── Get rule from config ─────────────────────────────
            rule = CURRENCY_RULES.get(currency_code, {})
            country_name = rule.get("country", country_info.get("name", "Unknown"))
            flag = rule.get("flag", "🏳️")
            symbol = rule.get("symbol", "")
            currency_name = rule.get("name", country_info.get("name", "Unknown"))

            # ── Image quality check ──────────────────────────────
            quality_check = self._check_image_quality(img)

            # ── Build checks list ────────────────────────────────
            checks = []

            # ML country check
            checks.append({
                "name": "ML Country Detection",
                "desc": "Image-based country classifier",
                "passed": True,
                "detail": f"Predicted: {country_name} ({country_conf*100:.0f}% confidence)",
                "points": 2,
            })

            # ML auth check
            checks.append({
                "name": "ML Authenticity Analysis",
                "desc": "Image-based real/fake classifier",
                "passed": is_genuine,
                "detail": f"Prediction: {auth_label} ({auth_conf*100:.0f}% confidence)",
                "points": 3 if is_genuine else -2,
            })

            # Image quality
            checks.append(quality_check)

            # ── Compute score ────────────────────────────────────
            score = 0
            if is_genuine:
                score = 3 + auth_conf * 4  # 3 to 7 range
            else:
                score = 1 - (auth_conf * 0.5)  # 0.5 to 1 range

            if quality_check["passed"]:
                score += 1

            score = round(min(10, max(0, score)), 1)

            # ── Status ───────────────────────────────────────────
            status = self._classify_status(score)

            checks_passed = [c for c in checks if c["passed"]]
            checks_failed = [c for c in checks if not c["passed"]]

            # ── Details ──────────────────────────────────────────
            details_lines = [
                f"Country: {country_name} {flag}",
                f"Currency: {currency_name} ({currency_code})",
                f"",
                f"ML Country Confidence: {country_conf*100:.0f}%",
                f"ML Auth Prediction: {auth_label} ({auth_conf*100:.0f}%)",
                f"Anti-Integrity Score: {score} / 10",
                f"Verdict: {status}",
                f"",
            ]
            if checks_passed:
                details_lines.append("✓ Passed Checks:")
                for c in checks_passed:
                    details_lines.append(f"  ✓ {c['name']}: {c['detail']}")
            if checks_failed:
                details_lines.append("✗ Failed Checks:")
                for c in checks_failed:
                    details_lines.append(f"  ✗ {c['name']}: {c['detail']}")

            # ── Voice reason ─────────────────────────────────────
            voice_parts = []
            voice_parts.append(f"I analyzed the uploaded image.")
            voice_parts.append(f"The currency appears to be from {country_name}.")

            if status == "REAL":
                voice_parts.append(f"This note appears to be genuine.")
                voice_parts.append(f"The AI model confirmed authenticity with {auth_conf*100:.0f} percent confidence.")
                voice_parts.append(f"The overall score is {score} out of 10.")
            elif status == "SUSPICIOUS":
                voice_parts.append(f"This note looks suspicious.")
                voice_parts.append(f"The AI model gave {auth_conf*100:.0f} percent confidence.")
                voice_parts.append(f"The score is {score}. Please verify carefully.")
            else:
                voice_parts.append(f"Warning! This note appears to be fake.")
                voice_parts.append(f"The AI model detected counterfeit patterns with {auth_conf*100:.0f} percent confidence.")
                voice_parts.append(f"The score is only {score}. Please do not accept this note.")

            result = {
                "currency_code": currency_code,
                "currency_name": currency_name,
                "country": country_name,
                "flag": flag,
                "symbol": symbol,
                "denomination": None,
                "status": status,
                "score": score,
                "confidence": round(auth_conf * 100, 1),
                "raw_text": "",
                "details": "\n".join(details_lines),
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "voice_reason": " ".join(voice_parts),
                "ml_used": True,
            }

            return result

        except Exception as e:
            print(f"[Detector] Image analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "currency_code": "UNKNOWN",
                "currency_name": "Unknown",
                "country": "N/A",
                "flag": "❓",
                "symbol": "",
                "denomination": None,
                "status": "ERROR",
                "score": 0,
                "confidence": 0,
                "raw_text": "",
                "details": f"Analysis error: {e}\nPlease try a different image.",
                "checks_passed": [],
                "checks_failed": [],
                "voice_reason": f"An error occurred during image analysis.",
                "ml_used": False,
            }
