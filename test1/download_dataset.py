"""
Download real currency note images using Wikimedia API + Google search fallback.
Creates 20 real + 20 fake (degraded) per country for training.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import urllib.request
import urllib.error
import ssl

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dataset", "currency_images")
IMG_SIZE = (224, 224)
IMAGES_PER_CLASS = 20

# Disable SSL verification for educational downloads
ssl._create_default_https_context = ssl._create_unverified_context

# ── Search terms for each currency ───────────────────────────────
CURRENCY_SEARCH = {
    "INR": ["Indian Rupee banknote", "India 500 rupee note", "India 100 rupee note",
            "Reserve Bank India currency", "India 2000 rupee", "India 200 rupee note",
            "India 50 rupee note", "Mahatma Gandhi currency note"],
    "USD": ["US Dollar bill", "United States 20 dollar", "US 100 dollar note",
            "Federal Reserve Note", "US 50 dollar bill", "US 5 dollar bill",
            "US 10 dollar bill", "American currency note"],
    "EUR": ["Euro banknote", "50 Euro note", "20 Euro note", "100 Euro note",
            "10 Euro note", "5 Euro note", "European Central Bank note"],
    "GBP": ["British Pound note", "Bank of England 20 pound", "UK 10 pound note",
            "British 50 pound note", "UK 5 pound note", "Sterling banknote"],
    "NPR": ["Nepal Rupee banknote", "Nepal Rastra Bank note", "Nepal 100 rupee",
            "Nepal 500 rupee note", "Nepal 1000 rupee note", "Nepalese currency"],
    "JPY": ["Japanese Yen banknote", "Japan 10000 yen", "Japan 5000 yen note",
            "Japan 1000 yen note", "Nippon Ginko note"],
    "CNY": ["Chinese Yuan banknote", "China 100 yuan note", "Renminbi note",
            "China 50 yuan", "Peoples Bank China currency"],
    "CAD": ["Canadian Dollar banknote", "Bank of Canada 20 dollar",
            "Canada 50 dollar note", "Canadian polymer note", "Canada 100 dollar"],
    "AUD": ["Australian Dollar banknote", "Australia 50 dollar note",
            "Australia polymer note", "Australian 20 dollar", "Reserve Bank Australia"],
    "RUB": ["Russian Ruble banknote", "Russia 1000 ruble", "Russia 5000 ruble note",
            "Bank of Russia currency", "Russian 500 ruble"],
    "BRL": ["Brazilian Real banknote", "Brazil 50 real note", "Brazil 100 real",
            "Banco Central Brasil currency", "Brazilian 20 real"],
    "KRW": ["South Korean Won banknote", "Korea 10000 won", "Korea 50000 won note",
            "Bank of Korea currency", "Korean 5000 won"],
    "SAR": ["Saudi Riyal banknote", "Saudi Arabia currency note", "Saudi 100 riyal",
            "Saudi Arabian Monetary note", "Saudi 500 riyal"],
    "AED": ["UAE Dirham banknote", "United Arab Emirates currency",
            "UAE 100 dirham note", "Central Bank UAE note", "UAE 500 dirham"],
    "THB": ["Thai Baht banknote", "Thailand 1000 baht", "Thailand 100 baht note",
            "Bank of Thailand currency", "Thai 500 baht"],
    "LKR": ["Sri Lanka Rupee banknote", "Sri Lanka 1000 rupee",
            "Central Bank Sri Lanka note", "Sri Lanka 500 rupee", "Ceylon currency"],
    "BDT": ["Bangladesh Taka banknote", "Bangladesh 500 taka",
            "Bangladesh Bank note", "Bangladesh 1000 taka", "Bangladeshi currency"],
}


def download_with_urllib(url, save_path, timeout=15):
    """Download image using urllib with proper headers."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        img_bytes = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False

        img = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        return False


def search_wikimedia_images(query, count=5):
    """Search Wikimedia Commons API for images."""
    urls = []
    try:
        search_q = urllib.parse.quote(query)
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?"
            f"action=query&list=search&srsearch={search_q}"
            f"&srnamespace=6&srlimit={count}&format=json"
        )
        req = urllib.request.Request(api_url, headers={
            "User-Agent": "CurrencyDatasetBot/1.0 (Educational project)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                title = item["title"]
                # Get actual image URL
                img_url = get_wikimedia_image_url(title)
                if img_url:
                    urls.append(img_url)
    except Exception as e:
        pass
    return urls


def get_wikimedia_image_url(title, width=400):
    """Get direct image URL from Wikimedia page title."""
    try:
        clean_title = urllib.parse.quote(title)
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?"
            f"action=query&titles={clean_title}"
            f"&prop=imageinfo&iiprop=url&iiurlwidth={width}&format=json"
        )
        req = urllib.request.Request(api_url, headers={
            "User-Agent": "CurrencyDatasetBot/1.0 (Educational project)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if "imageinfo" in page:
                info = page["imageinfo"][0]
                return info.get("thumburl") or info.get("url")
    except Exception:
        pass
    return None


def create_augmented_variants(img, count=20):
    """Create augmented variants from a single image."""
    variants = []
    for i in range(count):
        aug = img.copy()

        # Random horizontal flip
        if np.random.random() > 0.5:
            aug = cv2.flip(aug, 1)

        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.randint(-15, 15)
        aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)

        # Random rotation
        angle = np.random.uniform(-5, 5)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random slight crop
        if np.random.random() > 0.5:
            crop = np.random.randint(3, 12)
            aug = aug[crop:h-crop, crop:w-crop]
            aug = cv2.resize(aug, IMG_SIZE)

        # Random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        variants.append(aug)
    return variants


def create_fake_variants(img, count=20):
    """Create degraded/fake-looking variants."""
    fakes = []
    for i in range(count):
        fake = img.copy()

        # Strong colour distortion
        shift = np.random.randint(30, 80)
        channel = np.random.randint(0, 3)
        fake[:, :, channel] = np.clip(
            fake[:, :, channel].astype(np.int16) + shift, 0, 255
        ).astype(np.uint8)

        # Heavy blur
        k = np.random.choice([5, 7, 9, 11])
        fake = cv2.GaussianBlur(fake, (k, k), np.random.uniform(2, 5))

        # Heavy noise
        noise = np.random.normal(0, np.random.randint(15, 35), fake.shape).astype(np.int16)
        fake = np.clip(fake.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # JPEG artifacts
        quality = np.random.randint(5, 25)
        _, buf = cv2.imencode('.jpg', fake, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        fake = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # Large rotation
        angle = np.random.uniform(-12, 12)
        h, w = fake.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        fake = cv2.warpAffine(fake, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness distortion
        alpha = np.random.uniform(0.5, 1.5)
        beta = np.random.randint(-40, 40)
        fake = np.clip(alpha * fake + beta, 0, 255).astype(np.uint8)

        # Random patch erasure
        if np.random.random() > 0.5:
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            pw = np.random.randint(30, 70)
            ph = np.random.randint(30, 70)
            fake[y1:y1+ph, x1:x1+pw] = np.random.randint(0, 50, (ph, pw, 3), dtype=np.uint8)

        fake = cv2.resize(fake, IMG_SIZE)
        fakes.append(fake)
    return fakes


def main():
    import urllib.parse

    print("=" * 60)
    print("  CURRENCY IMAGE DOWNLOADER v2")
    print("  Using Wikimedia Commons API (public domain)")
    print(f"  {len(CURRENCY_SEARCH)} countries")
    print("=" * 60)

    total_real = 0
    total_fake = 0

    for code, queries in CURRENCY_SEARCH.items():
        real_dir = os.path.join(BASE_DIR, code, "real")
        fake_dir = os.path.join(BASE_DIR, code, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        print(f"\n  [{code}] Searching Wikimedia Commons...")

        # Collect image URLs from multiple searches
        all_urls = []
        for query in queries[:4]:  # limit to 4 queries
            urls = search_wikimedia_images(query, count=3)
            all_urls.extend(urls)
            time.sleep(0.5)

        # Remove duplicates
        all_urls = list(dict.fromkeys(all_urls))
        print(f"    Found {len(all_urls)} image URLs")

        # Download source images
        source_images = []
        for idx, url in enumerate(all_urls[:8]):  # max 8 source images
            temp_path = os.path.join(real_dir, f"_temp_{idx}.png")
            success = download_with_urllib(url, temp_path)
            if success:
                img = cv2.imread(temp_path)
                if img is not None:
                    source_images.append(img)
                    print(f"    [OK] Downloaded image {len(source_images)}")
                try:
                    os.remove(temp_path)
                except:
                    pass
            time.sleep(0.3)

        if not source_images:
            print(f"    [!] No images for {code} -- keeping existing synthetic data")
            existing = [f for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg'))]
            print(f"    Existing: {len(existing)} images")
            continue

        print(f"    Got {len(source_images)} source images")

        # Generate 20 real variants
        real_per_src = max(1, IMAGES_PER_CLASS // len(source_images))
        remainder = IMAGES_PER_CLASS - (real_per_src * len(source_images))

        real_count = 0
        for si, src in enumerate(source_images):
            n = real_per_src + (1 if si < remainder else 0)
            variants = create_augmented_variants(src, count=n)
            for v in variants:
                path = os.path.join(real_dir, f"{code}_real_{real_count+1:03d}.png")
                cv2.imwrite(path, v)
                real_count += 1
                if real_count >= IMAGES_PER_CLASS:
                    break
            if real_count >= IMAGES_PER_CLASS:
                break

        # Generate 20 fake variants
        fake_count = 0
        for si, src in enumerate(source_images):
            n = max(1, IMAGES_PER_CLASS // len(source_images))
            fakes = create_fake_variants(src, count=n)
            for f in fakes:
                path = os.path.join(fake_dir, f"{code}_fake_{fake_count+1:03d}.png")
                cv2.imwrite(path, f)
                fake_count += 1
                if fake_count >= IMAGES_PER_CLASS:
                    break
            if fake_count >= IMAGES_PER_CLASS:
                break

        total_real += real_count
        total_fake += fake_count
        print(f"    -> Saved {real_count} real + {fake_count} fake")

    print("\n" + "=" * 60)
    print(f"  DONE! {total_real} real + {total_fake} fake = {total_real + total_fake} total")
    print(f"  Saved to: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
