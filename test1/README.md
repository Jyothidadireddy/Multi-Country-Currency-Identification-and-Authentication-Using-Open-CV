# 💰 Multi-Country Currency Voice Assistant & Anti-Fake Detection System

## 1. Project Overview

The **Multi-Country Currency Voice Assistant** is an advanced, enterprise-grade machine learning and computer vision application designed to automatically detect, classify, and verify the authenticity of paper currency from 17 different countries. Aimed at solving the challenges of rapid cash handling, counterfeit prevention, and accessibility for the visually impaired, this project implements a highly optimized **Hybrid Detection Engine**. 

Rather than relying purely on classical rules or purely on deep learning, the system bridges both paradigms. It seamlessly merges **OCR (Optical Character Recognition)** rule-based pattern matching with a **Machine Learning Feature Extractor** (analyzing wavelet variance, image skewness, curtosis, and text entropy). Furthermore, it validates physical document integrity through sharpness and edge-density computer vision tests. 

Featuring a premium dark-glassmorphism Tkinter Graphical User Interface (GUI) and an asynchronous local Text-To-Speech (TTS) integration, the system serves as a professional, resilient, and responsive real-time tool for retail cashiers, banking institutions, and visually impaired individuals.

---

## 2. Core Features & Capabilities

### 2.1. Unprecedented Multi-Country Support
The system operates globally, recognizing the exact denomination and issuing country for 17 different major world currencies simultaneously without requiring the user to switch modes. The supported currencies are:
* **Americas:** US Dollar (USD), Canadian Dollar (CAD), Brazilian Real (BRL)
* **Europe:** Euro (EUR), British Pound (GBP), Russian Ruble (RUB)
* **Asia & Middle East:** Indian Rupee (INR), Japanese Yen (JPY), Chinese Yuan (CNY), South Korean Won (KRW), Saudi Riyal (SAR), UAE Dirham (AED)
* **South Asia (Extended):** Nepalese Rupee (NPR), Thai Baht (THB), Sri Lankan Rupee (LKR), Bangladeshi Taka (BDT)

### 2.2. The Hybrid Anti-Fake Engine (OCR + ML + CV)
Counterfeiters often manage to bypass one form of detection. To combat this, the system layers three distinct validation mechanisms:

1. **Semantic OCR Rule Matching:** Uses an intelligent word-boundary aware engine to find primary banking keywords (e.g., "FEDERAL RESERVE", "RESERVE BANK OF INDIA") and secondary security features (e.g., "GOVERNOR", "PROMISE TO PAY"). It employs an *Exclusion Map* to instantly disqualify false positives if colliding features from rival currencies appear.
2. **Machine Learning Model Analysis:** Trained on numerical feature representations (based on the UCI Banknote Authentication dataset), the system extracts continuous wavelet transform features from the captured image frame. A trained classifier (like Random Forest/CNN) then predicts whether the structural features map to genuine physical ink/paper characteristics with calculated confidence percentages.
3. **Computer Vision Integrity Checks:** It calculates Laplacian Variance to measure micro-blur (print sharpness) and applies Canny Edge Density algorithms. This prevents the system from being fooled by low-quality printed photocopies or low-DPI laser prints of currency. 

### 2.3. Intelligent Asynchronous Voice Feedback
Accessibility is a massive pillar of this physical detection project. A standard visually impaired individual requires real-time auditory confirmation. 
The system features an autonomous Text-to-Speech (TTS) narrator. To circumvent Python's Global Interpreter Lock (GIL) and Tkinter main-thread freezing, the voice synthesizer completely decouples into a discrete subprocess wrapper (`_speak.py`). It reads out:
* The detected denomination and country.
* The authenticity verdict: Genuine, Suspicious, or Fake.
* *Detailed reasoning* ("This note appears to be genuine. Here is why: official bank text was found, the print quality is good, and the trained AI model confirms genuineness with 92% confidence.").

### 2.4. Modern "Dark Glassmorphism" Tkinter UI
The application moves completely away from native grey-box GUI aesthetics by implementing a high-end styled layout with:
* **Interactive Live Camera Grid:** Beautiful bounding-box animations, real-time overlays, timestamps, and confidence percentages drawn right onto the video frame using OpenCV.
* **Verdict Analytics Panel:** Dedicated visual indicators for the currency flag, the Anti-Integrity Score, bar graphs showing ML/OCR confidence levels, and discrete sub-panels detailing precisely which security checks passed or failed.
* **Live Event Logging:** A chronological tracking feed detailing exactly what bills were scanned, helping tellers trace back transactions securely.

---

## 3. Deep Dive into the Architecture

The codebase handles heavy asynchronous machine learning without halting the display thread. Here is the module breakdown:

### `app.py` (The Interactive Client)
The entry point of the project. It launches the Tkinter `CurrencyApp` spanning the entire screen. The GUI has a 3-column architecture (Camera, Info, Log). `app.py` is responsible for initializing the local camera (`cv2.VideoCapture`), managing a specialized internal timer that batches OCR jobs at strategic intervals (to prevent GPU/CPU bottlenecking), and updating the graphical views via the `after()` event loop. Text overlay rendering on the live video happens dynamically in `_draw_overlay`.

### `detector.py` (The Brain)
Instances of `CurrencyDetector` manage the cognitive load. 
* It intercepts the raw bounds and confidence values from `EasyOCR`.
* Synthesizes a composite text string and sequentially iterates across `CURRENCY_RULES`.
* It evaluates Text Richness (denying notes that don't have enough geometric text clusters).
* Calculates an overarching `Anti-Integrity Score` from 1 to 10. `Score >= 3` guarantees Genuine, `Score >= 1.5` flags as Suspicious, and `Score < 1.5` results in a Fake warning.
* Implements the `get_stable_result()` function, meaning it enforces Temporal Frame Stabilization. The detector will silently track the past 5 video frames and will only declare a final decision to the GUI when it observes spatial and programmatic consistency. This prevents flickering or misfiring when a bill is quickly moved past the camera.

### `config.py` (The Ruleset Definition)
This file represents the global state definitions and constraints. The `CURRENCY_RULES` nested dictionary maps specific string tags and ISO currency codes to detection behaviors. The exclusion arrays prevent "FEDERAL RESERVE" and "BANGLADESH BANK" from cross-contaminating. Image quality constraints (e.g. `OCR_CONFIDENCE = 0.20`, downscaled to be inclusive of low-quality webcams) are tuned here.

### `voice.py` & `_speak.py` (The Audio Controller)
The `VoiceAssistant` acts as an auditory proxy. Because traditional `pyttsx3` halts local execution environments recursively, the architecture invokes Subprocess execution to spawn invisible isolated terminal instances targeting `_speak.py`. This guarantees real-time text narration without lagging the video framerate (locking it strictly to 30fps).

### `generate_dataset.py`, `download_dataset.py` & `features.py` (The Data Ops Layer)
These files act as the data pipeline backbone. They pull, augment, and manipulate original currency image datasets, generating degraded synthetic fakes through OpenCV filters to train the models robustness. `features.py` extracts the variance, skewness, and kurtosis to translate 2D pictorial layouts into 1D numerical data logic passable into machine learning nodes.

### `train_model.py` & `train_currency_model.py` (The ML Training Compilers)
These scripts handle dataset ingestion, normalisation (StandardScaler), model fitting, hyperparameter adjustments, and binary storage file exportation. When executed, they compile the final `.pkl` or `.h5` model files storing them natively in the `saved_models` directory where the main application will silently autoload them on boot up.

---

## 4. The Detection Workflow Pipeline

When a user shows a banknote to the webcam, here is the chronological pipeline execution happening under the hood in less than 300 milliseconds:

1. **Frame Capture:** The camera reads a 800x600 resolution pixel array matrix locally. 
2. **Quality Triaging:** Canny edge density evaluates if the target shape represents layered currency ink or a flat featureless plane.
3. **OCR Rasterisation:** `EasyOCR` maps spatial coordinates checking for English & localized script texts. 
4. **NLP Pattern Matching:** Detected strings run against the rule engine. Stop words and exclusions filter through the matrix to lock onto a primary identifier (e.g., finding "100" and "한국은행" isolates the asset as a 10000 South Korean Won candidate).
5. **Machine Learning Feature Feed:** The `features.py` intercepts the frame grid, calculates variance scalars, pushes them into `banknote_classifier.pkl` and waits for a boolean response (`0 = Genuine`, `1 = Fake`).
6. **Integrity Scoring Execution:** The system compiles the `score` metric.
7. **GUI Overlay Matrix Update:** The display thread updates the bounding boxes, turns the UI colors from grey to neon green, updates the Tkinter analytics progress bars, and writes a log into the array.
8. **Asynchronous Voicing:** The voice module triggers an explanation without breaking the video feed.

---

## 5. Setup & Installation

### Requirements
* **Operating System:** Windows 10/11, macOS, or Linux. (Optimized primarily for Windows COM TTS integrations).
* **Hardware:** Any 720p or 1080p Webcam. Highly recommended to use a Dedicated GPU (Nvidia CUDA) for native EasyOCR acceleration, though it fully supports CPU fallbacks.
* **Software:** Python 3.8 to 3.10.x.

### Library Dependencies
* `opencv-python`, `easyocr`: Vision and text extraction frameworks.
* `numpy`, `scikit-learn`: Data array operations and ML model algorithms.
* `pyttsx3`: Windows/Mac native offline voice synthesis.
* `Pillow (PIL)`: Specialized drawing onto the Tkinter canvas panels.
* `tkinter`: Standard GUI library.

### Initialization Plan

1. **Clone & Install the Repository**
   Ensure all base dependencies are mapped locally to the virtual environment. Use `pip install -r requirements.txt`.
2. **Pre-requisite Model Generation (IMPORTANT)**
   Because machine learning binaries are hardware-dependent, users must run the training scripts *before* launching the GUI to generate the `banknote_classifier.pkl` local files.
   ```bash
   python train_model.py
   ```
   *The system is engineered to fail gracefully. If you do not run the training model, the application will simply operate in "OCR Only Mode."*
3. **Execution**
   Launch the overarching Tkinter interface.
   ```bash
   python app.py
   ```
4. **Interacting**
   Click the **Start Camera** button. Place the physical currency directly in the center of the viewport, holding it steady for 1–2 seconds to allow the Temporal Frame Stabilization array to verify consistency. Turn the **Voice ON** to hear auditory output metrics.

---

## 6. Future Implementations & Scaleability 

This architecture was designed inherently to scale aggressively. Currently, the `config.py` acts as a static registry, but the backend is purely modular.
* To add a completely new foreign currency, a developer only needs to append a new JSON-like entry inside `CURRENCY_RULES` with localized text keywords. The detector instantly inherits support.
* Future integrations map naturally for mobile porting (converting the ML nodes to TensorFlow Lite protocols) and potentially upgrading the computer vision stack from EasyOCR sequentially tracking grids, to zero-shot bounding-box tracking engines like YOLOv8 for spatial framing in turbulent lighting. 
* Implementation of custom fine-tuned ViT (Vision Transformers) models can fully remove the need for OCR in severely degraded currency environments.

This application highlights an ideal marriage between accessibility-focused UI creation, localized offline voice synthesis, probabilistic OCR parsing algorithms, and computer-vision powered data-security validations.
