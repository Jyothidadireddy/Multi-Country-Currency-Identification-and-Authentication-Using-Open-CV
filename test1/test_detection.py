"""Quick test to verify multi-currency detection works correctly."""
from detector import CurrencyDetector

d = CurrencyDetector()

# --- Test 1: US Dollar ---
usd_ocr = [
    ([0,0,0,0], "FEDERAL RESERVE NOTE", 0.85),
    ([0,0,0,0], "THE UNITED STATES OF AMERICA", 0.80),
    ([0,0,0,0], "IN GOD WE TRUST", 0.75),
    ([0,0,0,0], "TWENTY DOLLARS", 0.70),
    ([0,0,0,0], "20", 0.90),
    ([0,0,0,0], "WASHINGTON", 0.65),
    ([0,0,0,0], "THIS NOTE IS LEGAL TENDER", 0.60),
    ([0,0,0,0], "SECRETARY OF THE TREASURY", 0.55),
]
result = d.analyze(usd_ocr)
print(f"USD test: {result['currency_code']} - {result['country']} - {result['status']}")
print(f"  Score: {result['score']}, Confidence: {result['confidence']}")
d.reset()

# --- Test 2: Indian Rupee ---
inr_ocr = [
    ([0,0,0,0], "RESERVE BANK OF INDIA", 0.85),
    ([0,0,0,0], "BHARATIYA", 0.80),
    ([0,0,0,0], "MAHATMA GANDHI", 0.75),
    ([0,0,0,0], "500", 0.90),
    ([0,0,0,0], "GOVERNOR", 0.70),
    ([0,0,0,0], "GUARANTEE", 0.65),
    ([0,0,0,0], "RBI", 0.60),
    ([0,0,0,0], "PAYABLE TO THE BEARER", 0.55),
]
result2 = d.analyze(inr_ocr)
print(f"INR test: {result2['currency_code']} - {result2['country']} - {result2['status']}")
print(f"  Score: {result2['score']}, Confidence: {result2['confidence']}")
d.reset()

# --- Test 3: Euro ---
eur_ocr = [
    ([0,0,0,0], "EUROPEAN CENTRAL BANK", 0.85),
    ([0,0,0,0], "EURO", 0.80),
    ([0,0,0,0], "BCE", 0.75),
    ([0,0,0,0], "ECB", 0.70),
    ([0,0,0,0], "50", 0.90),
    ([0,0,0,0], "EUROSYSTEM", 0.65),
]
result3 = d.analyze(eur_ocr)
print(f"EUR test: {result3['currency_code']} - {result3['country']} - {result3['status']}")
print(f"  Score: {result3['score']}, Confidence: {result3['confidence']}")
d.reset()

# --- Test 4: British Pound ---
gbp_ocr = [
    ([0,0,0,0], "BANK OF ENGLAND", 0.85),
    ([0,0,0,0], "POUND", 0.80),
    ([0,0,0,0], "PROMISE TO PAY", 0.75),
    ([0,0,0,0], "ELIZABETH", 0.70),
    ([0,0,0,0], "20", 0.90),
]
result4 = d.analyze(gbp_ocr)
print(f"GBP test: {result4['currency_code']} - {result4['country']} - {result4['status']}")
print(f"  Score: {result4['score']}, Confidence: {result4['confidence']}")
d.reset()

# --- Test 5: Nepal ---
npr_ocr = [
    ([0,0,0,0], "NEPAL RASTRA BANK", 0.85),
    ([0,0,0,0], "NEPAL", 0.80),
    ([0,0,0,0], "100", 0.75),
    ([0,0,0,0], "GOVERNOR", 0.70),
    ([0,0,0,0], "NEPALESE", 0.65),
]
result5 = d.analyze(npr_ocr)
print(f"NPR test: {result5['currency_code']} - {result5['country']} - {result5['status']}")
print(f"  Score: {result5['score']}, Confidence: {result5['confidence']}")
d.reset()

print("\n=== All tests complete ===")
