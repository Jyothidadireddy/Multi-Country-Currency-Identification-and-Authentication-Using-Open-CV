"""
╔══════════════════════════════════════════════════════════════════╗
║   CURRENCY VOICE ASSISTANT — Premium Tkinter Application       ║
║   Live camera · Multi-country detection · Anti-fake scoring    ║
║   Voice feedback · Dark glassmorphism theme                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import easyocr
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from datetime import datetime

from config import CURRENCY_RULES, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
from detector import CurrencyDetector
from voice import VoiceAssistant


# ═══════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (Dark Glassmorphism)
# ═══════════════════════════════════════════════════════════════════
C = {
    "bg":           "#0d1117",
    "bg_card":      "#161b22",
    "bg_panel":     "#1c2333",
    "border":       "#30363d",
    "accent":       "#58a6ff",
    "accent2":      "#7ee787",
    "warning":      "#f0883e",
    "danger":       "#f85149",
    "text":         "#e6edf3",
    "text_dim":     "#8b949e",
    "real":         "#2ea043",
    "suspicious":   "#d29922",
    "fake":         "#f85149",
    "unknown":      "#484f58",
}


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════
class CurrencyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💰 Currency Voice Assistant — Anti-Fake Detection System")
        self.root.configure(bg=C["bg"])
        self.root.state("zoomed")  # maximise on Windows
        self.root.minsize(1200, 700)

        # ── State ────────────────────────────────────────────────
        self.camera = None
        self.camera_running = False
        self.ocr_reader = None
        self.ocr_busy = False
        self.detector = CurrencyDetector()
        self.voice = VoiceAssistant()
        self.scan_log = []          # history of detections
        self.current_result = None
        self.frame_count = 0
        self.ocr_interval = 15      # run OCR every N frames (perf)

        # ── Build GUI ────────────────────────────────────────────
        self._build_styles()
        self._build_header()
        self._build_body()
        self._build_footer()

        # ── Load OCR in background ──────────────────────────────
        self._status("Initialising OCR engine …")
        threading.Thread(target=self._init_ocr, daemon=True).start()

        # ── Cleanup on close ────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════════════════════════════════════
    #  STYLES
    # ═══════════════════════════════════════════════════════════════
    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=C["bg"])
        style.configure("Card.TFrame", background=C["bg_card"])
        style.configure("Panel.TFrame", background=C["bg_panel"])

        style.configure("Header.TLabel", background=C["bg"],
                        foreground=C["accent"], font=("Segoe UI", 18, "bold"))
        style.configure("Sub.TLabel", background=C["bg"],
                        foreground=C["text_dim"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=C["bg_card"],
                        foreground=C["text"], font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", background=C["bg_card"],
                        foreground=C["accent"], font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", background=C["bg"],
                        foreground=C["text_dim"], font=("Segoe UI", 9))
        style.configure("Big.TLabel", background=C["bg_card"],
                        foreground=C["text"], font=("Segoe UI", 26, "bold"))
        style.configure("Flag.TLabel", background=C["bg_card"],
                        foreground=C["text"], font=("Segoe UI", 40))

        # Buttons
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"),
                        padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", C["accent"]), ("!active", C["bg_panel"])],
                  foreground=[("active", C["bg"]), ("!active", C["text"])])

        style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"),
                        padding=(16, 8))
        style.map("Danger.TButton",
                  background=[("active", C["danger"]), ("!active", C["bg_panel"])],
                  foreground=[("active", C["bg"]), ("!active", C["danger"])])

    # ═══════════════════════════════════════════════════════════════
    #  HEADER
    # ═══════════════════════════════════════════════════════════════
    def _build_header(self):
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill="x", padx=20, pady=(15, 5))

        left = ttk.Frame(header, style="Dark.TFrame")
        left.pack(side="left")

        ttk.Label(left, text="💰 Currency Voice Assistant",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(left, text="Real-time multi-country detection  ·  ML + OCR hybrid  ·  Anti-fake scoring  ·  Voice feedback",
                  style="Sub.TLabel").pack(anchor="w")

        right = ttk.Frame(header, style="Dark.TFrame")
        right.pack(side="right")

        self.btn_camera = tk.Button(
            right, text="▶  Start Camera", font=("Segoe UI", 11, "bold"),
            bg=C["accent"], fg=C["bg"], activebackground="#4090e0",
            relief="flat", padx=18, pady=8, cursor="hand2",
            command=self._toggle_camera)
        self.btn_camera.pack(side="left", padx=5)

        self.btn_voice = tk.Button(
            right, text="🔊 Voice ON", font=("Segoe UI", 11, "bold"),
            bg=C["bg_panel"], fg=C["accent2"], activebackground=C["bg_card"],
            relief="flat", padx=18, pady=8, cursor="hand2",
            command=self._toggle_voice)
        self.btn_voice.pack(side="left", padx=5)

        self.btn_reset = tk.Button(
            right, text="🗑  Clear", font=("Segoe UI", 11, "bold"),
            bg=C["bg_panel"], fg=C["warning"], activebackground=C["bg_card"],
            relief="flat", padx=18, pady=8, cursor="hand2",
            command=self._reset)
        self.btn_reset.pack(side="left", padx=5)

        self.btn_other_country = tk.Button(
            right, text="🌍 Other Country", font=("Segoe UI", 11, "bold"),
            bg="#7c3aed", fg="#ffffff", activebackground="#6d28d9",
            relief="flat", padx=18, pady=8, cursor="hand2",
            command=self._open_upload_window)
        self.btn_other_country.pack(side="left", padx=5)

        # Separator
        sep = tk.Frame(self.root, bg=C["border"], height=1)
        sep.pack(fill="x", padx=20, pady=5)

    # ═══════════════════════════════════════════════════════════════
    #  BODY  (3-column layout)
    # ═══════════════════════════════════════════════════════════════
    def _build_body(self):
        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill="both", expand=True, padx=20, pady=5)
        body.columnconfigure(0, weight=3, minsize=500)
        body.columnconfigure(1, weight=2, minsize=320)
        body.columnconfigure(2, weight=2, minsize=280)

        self._build_camera_panel(body)
        self._build_result_panel(body)
        self._build_log_panel(body)

    # ── Camera Panel ─────────────────────────────────────────────
    def _build_camera_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                        highlightthickness=1, padx=2, pady=2)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)

        title_bar = tk.Frame(card, bg=C["bg_card"])
        title_bar.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(title_bar, text="📷  Live Camera Feed", font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg=C["accent"]).pack(side="left")

        self.camera_status = tk.Label(title_bar, text="● OFFLINE", font=("Segoe UI", 10, "bold"),
                                      bg=C["bg_card"], fg=C["danger"])
        self.camera_status.pack(side="right")

        # Canvas for video
        self.canvas = tk.Canvas(card, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Placeholder text on canvas
        self.canvas.update_idletasks()
        self.canvas.create_text(
            400, 250, text="Press ▶ Start Camera to begin scanning",
            fill=C["text_dim"], font=("Segoe UI", 14), tags="placeholder")

    # ── Result Panel ─────────────────────────────────────────────
    def _build_result_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                        highlightthickness=1, padx=2, pady=2)
        card.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        tk.Label(card, text="🔍  Detection Result", font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", padx=14, pady=(12, 4))

        sep = tk.Frame(card, bg=C["border"], height=1)
        sep.pack(fill="x", padx=14, pady=4)

        # Flag / country
        self.lbl_flag = tk.Label(card, text="❓", font=("Segoe UI", 48),
                                  bg=C["bg_card"], fg=C["text"])
        self.lbl_flag.pack(pady=(10, 0))

        self.lbl_country = tk.Label(card, text="No Detection Yet",
                                     font=("Segoe UI", 16, "bold"),
                                     bg=C["bg_card"], fg=C["text"])
        self.lbl_country.pack(pady=2)

        # Denomination
        self.lbl_denom = tk.Label(card, text="—",
                                   font=("Segoe UI", 28, "bold"),
                                   bg=C["bg_card"], fg=C["accent"])
        self.lbl_denom.pack(pady=6)

        # Status badge
        self.status_frame = tk.Frame(card, bg=C["bg_card"])
        self.status_frame.pack(pady=6)

        self.lbl_status = tk.Label(self.status_frame, text=" NO DATA ",
                                    font=("Segoe UI", 14, "bold"),
                                    bg=C["unknown"], fg="#ffffff", padx=20, pady=6)
        self.lbl_status.pack()

        # Details
        sep2 = tk.Frame(card, bg=C["border"], height=1)
        sep2.pack(fill="x", padx=14, pady=8)

        self.lbl_details = tk.Label(card, text="Show a currency note to the camera\nto begin detection.",
                                     font=("Segoe UI", 10), bg=C["bg_card"],
                                     fg=C["text_dim"], justify="left", wraplength=280)
        self.lbl_details.pack(anchor="w", padx=14, pady=(0, 4))

        # Score bar
        score_frame = tk.Frame(card, bg=C["bg_card"])
        score_frame.pack(fill="x", padx=14, pady=(4, 10))

        tk.Label(score_frame, text="Anti-Integrity Score", font=("Segoe UI", 9),
                 bg=C["bg_card"], fg=C["text_dim"]).pack(anchor="w")

        self.score_canvas = tk.Canvas(score_frame, height=18, bg=C["bg_panel"],
                                       highlightthickness=0)
        self.score_canvas.pack(fill="x", pady=4)

        self.lbl_score_val = tk.Label(score_frame, text="0 / 10",
                                       font=("Segoe UI", 9, "bold"),
                                       bg=C["bg_card"], fg=C["text_dim"])
        self.lbl_score_val.pack(anchor="e")

        # Confidence
        conf_frame = tk.Frame(card, bg=C["bg_card"])
        conf_frame.pack(fill="x", padx=14, pady=(0, 14))

        tk.Label(conf_frame, text="OCR Confidence", font=("Segoe UI", 9),
                 bg=C["bg_card"], fg=C["text_dim"]).pack(anchor="w")

        self.conf_canvas = tk.Canvas(conf_frame, height=18, bg=C["bg_panel"],
                                      highlightthickness=0)
        self.conf_canvas.pack(fill="x", pady=4)

        self.lbl_conf_val = tk.Label(conf_frame, text="0%",
                                      font=("Segoe UI", 9, "bold"),
                                      bg=C["bg_card"], fg=C["text_dim"])
        self.lbl_conf_val.pack(anchor="e")

    # ── Log Panel ────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"], highlightbackground=C["border"],
                        highlightthickness=1, padx=2, pady=2)
        card.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=4)

        tk.Label(card, text="📝  Scan History", font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", padx=14, pady=(12, 4))

        sep = tk.Frame(card, bg=C["border"], height=1)
        sep.pack(fill="x", padx=14, pady=4)

        # Scrollable log
        log_container = tk.Frame(card, bg=C["bg_card"])
        log_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_container, bg=C["bg_panel"], fg=C["text"],
                                 font=("Consolas", 10), relief="flat",
                                 insertbackground=C["text"], wrap="word",
                                 state="disabled", padx=10, pady=8)
        scrollbar = ttk.Scrollbar(log_container, orient="vertical",
                                   command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Supported currencies summary at bottom
        summary_frame = tk.Frame(card, bg=C["bg_panel"], padx=10, pady=8)
        summary_frame.pack(fill="x", padx=10, pady=(0, 10))

        flags = " ".join(r["flag"] for r in CURRENCY_RULES.values())
        tk.Label(summary_frame, text="Supported Currencies", font=("Segoe UI", 9, "bold"),
                 bg=C["bg_panel"], fg=C["accent"]).pack(anchor="w")
        tk.Label(summary_frame, text=flags, font=("Segoe UI", 14),
                 bg=C["bg_panel"], fg=C["text"]).pack(anchor="w", pady=2)
        ml_status = "ML Model Active ✓" if self.detector.ml_available else "OCR Only (run train_model.py)"
        tk.Label(summary_frame,
                 text=f"{len(CURRENCY_RULES)} countries  ·  {ml_status}  ·  Anti-fake scoring",
                 font=("Segoe UI", 8), bg=C["bg_panel"], fg=C["text_dim"]).pack(anchor="w")

    # ═══════════════════════════════════════════════════════════════
    #  FOOTER
    # ═══════════════════════════════════════════════════════════════
    def _build_footer(self):
        footer = tk.Frame(self.root, bg=C["bg"], height=28)
        footer.pack(fill="x", padx=20, pady=(2, 8))

        self.lbl_status_bar = tk.Label(footer, text="Ready",
                                        font=("Segoe UI", 9),
                                        bg=C["bg"], fg=C["text_dim"])
        self.lbl_status_bar.pack(side="left")

        ml_tag = "ML + OCR Hybrid" if self.detector.ml_available else "OCR Only"
        tk.Label(footer, text=f"Currency Voice Assistant v3.0  ·  {ml_tag}  ·  pyttsx3",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["text_dim"]).pack(side="right")

    # ═══════════════════════════════════════════════════════════════
    #  OCR INIT
    # ═══════════════════════════════════════════════════════════════
    def _init_ocr(self):
        try:
            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._status("OCR engine ready")
        except Exception as e:
            self._status(f"OCR init error: {e}")

    # ═══════════════════════════════════════════════════════════════
    #  CAMERA CONTROLS
    # ═══════════════════════════════════════════════════════════════
    def _toggle_camera(self):
        if self.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        try:
            self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            if not self.camera.isOpened():
                raise RuntimeError("Cannot open camera")

            self.camera_running = True
            self.btn_camera.configure(text="⏹  Stop Camera", bg=C["danger"])
            self.camera_status.configure(text="● LIVE", fg=C["real"])
            self.canvas.delete("placeholder")
            self._status("Camera started — show a currency note")
            self._update_frame()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self._status(f"Camera error: {e}")

    def _stop_camera(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.btn_camera.configure(text="▶  Start Camera", bg=C["accent"])
        self.camera_status.configure(text="● OFFLINE", fg=C["danger"])
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 250, text="Camera stopped. Press ▶ to resume.",
            fill=C["text_dim"], font=("Segoe UI", 14), tags="placeholder")
        self._status("Camera stopped")

    # ═══════════════════════════════════════════════════════════════
    #  FRAME UPDATE LOOP
    # ═══════════════════════════════════════════════════════════════
    def _update_frame(self):
        if not self.camera_running:
            return

        ret, frame = self.camera.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        self.frame_count += 1

        # Run OCR periodically in a separate thread
        if self.frame_count % self.ocr_interval == 0 and not self.ocr_busy and self.ocr_reader:
            self.ocr_busy = True
            frame_copy = frame.copy()
            threading.Thread(target=self._run_ocr, args=(frame_copy,), daemon=True).start()

        # Draw overlay on frame
        display = self._draw_overlay(frame)

        # Convert to Tkinter image
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit to canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            img = img.resize((cw, ch), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas._imgtk = imgtk  # prevent garbage collection

        self.root.after(33, self._update_frame)  # ~30 fps

    # ═══════════════════════════════════════════════════════════════
    #  OCR PROCESSING (background thread)
    # ═══════════════════════════════════════════════════════════════
    def _run_ocr(self, frame):
        try:
            results = self.ocr_reader.readtext(frame)
            # Pass frame to detector for ML + image quality analysis
            result = self.detector.analyze(results, frame=frame)
            stable = self.detector.get_stable_result()

            if stable and stable["currency_code"] != "UNKNOWN":
                self.current_result = stable
                self.voice.announce_result(stable)
                # Schedule GUI update on main thread
                self.root.after(0, lambda: self._update_result_panel(stable))
            elif result:
                self.current_result = result
                self.root.after(0, lambda: self._update_result_panel(result))
        except Exception as e:
            print(f"[OCR Error] {e}")
        finally:
            self.ocr_busy = False

    # ═══════════════════════════════════════════════════════════════
    #  DRAW HUD OVERLAY ON VIDEO FRAME
    # ═══════════════════════════════════════════════════════════════
    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (13, 17, 23), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(frame, "CURRENCY SCANNER", (15, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 120, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 148, 158), 1)

        # Detection result overlay
        if self.current_result and self.current_result["currency_code"] != "UNKNOWN":
            r = self.current_result
            status = r["status"]
            colour = (42, 160, 67) if status == "REAL" else \
                     (34, 153, 210) if status == "SUSPICIOUS" else (248, 81, 73)

            # Bottom bar
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 70), (w, h), (22, 27, 34), -1)
            frame = cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0)

            label = f"{r['flag']} {r['currency_code']}"
            if r["denomination"]:
                label += f" {r['symbol']}{r['denomination']}"
            label += f"  |  {status}"

            cv2.putText(frame, label, (15, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            score_text = f"Score: {r['score']}  |  Conf: {r['confidence']}%"
            cv2.putText(frame, score_text, (15, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)

            # Scanning border
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), colour, 3)

        else:
            # Scanning animation border
            t = int(time.time() * 1000) % 2000
            alpha = abs(t - 1000) / 1000.0
            colour = (int(88 * alpha), int(166 * alpha), int(255 * alpha))
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), colour, 2)

        return frame

    # ═══════════════════════════════════════════════════════════════
    #  UPDATE RESULT PANEL
    # ═══════════════════════════════════════════════════════════════
    def _update_result_panel(self, result):
        r = result

        # Flag
        self.lbl_flag.configure(text=r["flag"])

        # Country
        self.lbl_country.configure(text=f"{r['country']}  ({r['currency_code']})")

        # Denomination
        if r["denomination"]:
            self.lbl_denom.configure(text=f"{r['symbol']}{r['denomination']}")
        else:
            self.lbl_denom.configure(text=r["currency_name"])

        # Status badge
        status = r["status"]
        if status == "REAL":
            bg, label = C["real"], "✓  GENUINE"
        elif status == "SUSPICIOUS":
            bg, label = C["suspicious"], "⚠  SUSPICIOUS"
        elif status == "FAKE":
            bg, label = C["fake"], "✗  FAKE"
        else:
            bg, label = C["unknown"], "—  NO DATA"
        self.lbl_status.configure(text=f"  {label}  ", bg=bg)

        # Details
        self.lbl_details.configure(text=r["details"])

        # Score bar
        self._draw_bar(self.score_canvas, r["score"], 10,
                       C["real"] if status == "REAL" else
                       C["suspicious"] if status == "SUSPICIOUS" else C["fake"])
        self.lbl_score_val.configure(text=f"{r['score']} / 10")

        # Confidence bar
        self._draw_bar(self.conf_canvas, r["confidence"], 100, C["accent"])
        self.lbl_conf_val.configure(text=f"{r['confidence']}%")

        # Add to log
        self._add_log_entry(r)

    def _draw_bar(self, canvas, value, max_val, colour):
        canvas.delete("all")
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 5:
            return
        fill_w = max(0, int((value / max(max_val, 1)) * w))
        canvas.create_rectangle(0, 0, fill_w, h, fill=colour, outline="")

    # ═══════════════════════════════════════════════════════════════
    #  SCAN LOG
    # ═══════════════════════════════════════════════════════════════
    def _add_log_entry(self, result):
        # Avoid duplicate consecutive logs
        sig = f"{result['currency_code']}_{result['denomination']}_{result['status']}"
        if self.scan_log and self.scan_log[-1] == sig:
            return
        self.scan_log.append(sig)

        ts = datetime.now().strftime("%H:%M:%S")
        status_icon = "✓" if result["status"] == "REAL" else \
                      "⚠" if result["status"] == "SUSPICIOUS" else "✗"

        entry = (f"[{ts}] {status_icon} {result['flag']} {result['currency_code']}"
                 f" {result['symbol']}{result['denomination'] or '?'}"
                 f"  →  {result['status']}\n")

        self.log_text.configure(state="normal")
        self.log_text.insert("end", entry)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════════
    #  CONTROLS
    # ═══════════════════════════════════════════════════════════════
    def _toggle_voice(self):
        enabled = self.voice.toggle()
        if enabled:
            self.btn_voice.configure(text="🔊 Voice ON", fg=C["accent2"])
            self._status("Voice assistant enabled")
        else:
            self.btn_voice.configure(text="🔇 Voice OFF", fg=C["text_dim"])
            self._status("Voice assistant disabled")

    def _reset(self):
        self.detector.reset()
        self.current_result = None
        self.scan_log.clear()

        self.lbl_flag.configure(text="❓")
        self.lbl_country.configure(text="No Detection Yet")
        self.lbl_denom.configure(text="—")
        self.lbl_status.configure(text=" NO DATA ", bg=C["unknown"])
        self.lbl_details.configure(text="Show a currency note to the camera\nto begin detection.")
        self.score_canvas.delete("all")
        self.conf_canvas.delete("all")
        self.lbl_score_val.configure(text="0 / 10")
        self.lbl_conf_val.configure(text="0%")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._status("Detection reset")

    def _status(self, msg):
        try:
            self.lbl_status_bar.configure(text=msg)
        except Exception:
            pass

    def _on_close(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.voice.shutdown()
        self.root.destroy()

    def _open_upload_window(self):
        """Open the image upload detection window."""
        UploadWindow(self.root, self.detector, self.voice)


# ═══════════════════════════════════════════════════════════════════
#  UPLOAD DETECTION WINDOW
# ═══════════════════════════════════════════════════════════════════
class UploadWindow:
    """Standalone window for upload-based currency detection (US/UK)."""

    def __init__(self, parent, detector, voice):
        self.detector = detector
        self.voice = voice
        self.current_result = None
        self.scan_log = []

        # ── Create Toplevel window ───────────────────────────────
        self.win = tk.Toplevel(parent)
        self.win.title("🌍 Other Country Detection — Upload Image")
        self.win.configure(bg=C["bg"])
        self.win.geometry("1100x700")
        self.win.minsize(900, 600)

        # Try to center on parent
        try:
            px = parent.winfo_x() + 50
            py = parent.winfo_y() + 50
            self.win.geometry(f"+{px}+{py}")
        except Exception:
            pass

        self._build_header()
        self._build_body()
        self._build_footer()

    # ═══════════════════════════════════════════════════════════════
    #  HEADER
    # ═══════════════════════════════════════════════════════════════
    def _build_header(self):
        header = tk.Frame(self.win, bg=C["bg"])
        header.pack(fill="x", padx=20, pady=(15, 5))

        left = tk.Frame(header, bg=C["bg"])
        left.pack(side="left")

        tk.Label(left, text="🌍 Other Country Detection",
                 font=("Segoe UI", 18, "bold"),
                 bg=C["bg"], fg="#a78bfa").pack(anchor="w")
        tk.Label(left, text="Upload US/UK currency images  ·  ML image analysis  ·  Real/Fake detection  ·  Voice feedback",
                 font=("Segoe UI", 10),
                 bg=C["bg"], fg=C["text_dim"]).pack(anchor="w")

        right = tk.Frame(header, bg=C["bg"])
        right.pack(side="right")

        self.btn_upload = tk.Button(
            right, text="📁  Upload Image", font=("Segoe UI", 12, "bold"),
            bg="#7c3aed", fg="#ffffff", activebackground="#6d28d9",
            relief="flat", padx=22, pady=10, cursor="hand2",
            command=self._upload_image)
        self.btn_upload.pack(side="left", padx=5)

        self.btn_clear = tk.Button(
            right, text="🗑  Clear", font=("Segoe UI", 11, "bold"),
            bg=C["bg_panel"], fg=C["warning"], activebackground=C["bg_card"],
            relief="flat", padx=18, pady=8, cursor="hand2",
            command=self._reset)
        self.btn_clear.pack(side="left", padx=5)

        # Separator
        sep = tk.Frame(self.win, bg=C["border"], height=1)
        sep.pack(fill="x", padx=20, pady=5)

    # ═══════════════════════════════════════════════════════════════
    #  BODY (3-column layout)
    # ═══════════════════════════════════════════════════════════════
    def _build_body(self):
        body = tk.Frame(self.win, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=5)
        body.columnconfigure(0, weight=3, minsize=400)
        body.columnconfigure(1, weight=2, minsize=300)
        body.columnconfigure(2, weight=2, minsize=250)

        self._build_image_panel(body)
        self._build_result_panel(body)
        self._build_log_panel(body)

    # ── Image Preview Panel ──────────────────────────────────────
    def _build_image_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"],
                        highlightbackground=C["border"], highlightthickness=1,
                        padx=2, pady=2)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)

        title_bar = tk.Frame(card, bg=C["bg_card"])
        title_bar.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(title_bar, text="🖼️  Uploaded Image",
                 font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg="#a78bfa").pack(side="left")

        self.img_status = tk.Label(title_bar, text="● NO IMAGE",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=C["bg_card"], fg=C["text_dim"])
        self.img_status.pack(side="right")

        # Image canvas
        self.img_canvas = tk.Canvas(card, bg="#000000", highlightthickness=0)
        self.img_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Placeholder text
        self.img_canvas.update_idletasks()
        self.img_canvas.create_text(
            300, 200, text="Click 📁 Upload Image to select\na US or UK currency note image",
            fill=C["text_dim"], font=("Segoe UI", 14), tags="placeholder",
            justify="center")

    # ── Result Panel ─────────────────────────────────────────────
    def _build_result_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"],
                        highlightbackground=C["border"], highlightthickness=1,
                        padx=2, pady=2)
        card.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        tk.Label(card, text="🔍  Detection Result",
                 font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg="#a78bfa").pack(anchor="w", padx=14, pady=(12, 4))

        sep = tk.Frame(card, bg=C["border"], height=1)
        sep.pack(fill="x", padx=14, pady=4)

        # Flag
        self.lbl_flag = tk.Label(card, text="❓", font=("Segoe UI", 48),
                                  bg=C["bg_card"], fg=C["text"])
        self.lbl_flag.pack(pady=(10, 0))

        # Country
        self.lbl_country = tk.Label(card, text="No Detection Yet",
                                     font=("Segoe UI", 16, "bold"),
                                     bg=C["bg_card"], fg=C["text"])
        self.lbl_country.pack(pady=2)

        # Denomination / currency
        self.lbl_denom = tk.Label(card, text="—",
                                   font=("Segoe UI", 28, "bold"),
                                   bg=C["bg_card"], fg="#a78bfa")
        self.lbl_denom.pack(pady=6)

        # Status badge
        self.status_frame = tk.Frame(card, bg=C["bg_card"])
        self.status_frame.pack(pady=6)

        self.lbl_status = tk.Label(self.status_frame, text=" NO DATA ",
                                    font=("Segoe UI", 14, "bold"),
                                    bg=C["unknown"], fg="#ffffff",
                                    padx=20, pady=6)
        self.lbl_status.pack()

        # Details
        sep2 = tk.Frame(card, bg=C["border"], height=1)
        sep2.pack(fill="x", padx=14, pady=8)

        self.lbl_details = tk.Label(
            card, text="Upload a currency image to\nbegin detection.",
            font=("Segoe UI", 10), bg=C["bg_card"],
            fg=C["text_dim"], justify="left", wraplength=280)
        self.lbl_details.pack(anchor="w", padx=14, pady=(0, 4))

        # Score bar
        score_frame = tk.Frame(card, bg=C["bg_card"])
        score_frame.pack(fill="x", padx=14, pady=(4, 10))

        tk.Label(score_frame, text="Anti-Integrity Score",
                 font=("Segoe UI", 9),
                 bg=C["bg_card"], fg=C["text_dim"]).pack(anchor="w")

        self.score_canvas = tk.Canvas(score_frame, height=18,
                                      bg=C["bg_panel"], highlightthickness=0)
        self.score_canvas.pack(fill="x", pady=4)

        self.lbl_score_val = tk.Label(score_frame, text="0 / 10",
                                       font=("Segoe UI", 9, "bold"),
                                       bg=C["bg_card"], fg=C["text_dim"])
        self.lbl_score_val.pack(anchor="e")

        # Confidence bar
        conf_frame = tk.Frame(card, bg=C["bg_card"])
        conf_frame.pack(fill="x", padx=14, pady=(0, 14))

        tk.Label(conf_frame, text="ML Confidence",
                 font=("Segoe UI", 9),
                 bg=C["bg_card"], fg=C["text_dim"]).pack(anchor="w")

        self.conf_canvas = tk.Canvas(conf_frame, height=18,
                                      bg=C["bg_panel"], highlightthickness=0)
        self.conf_canvas.pack(fill="x", pady=4)

        self.lbl_conf_val = tk.Label(conf_frame, text="0%",
                                      font=("Segoe UI", 9, "bold"),
                                      bg=C["bg_card"], fg=C["text_dim"])
        self.lbl_conf_val.pack(anchor="e")

    # ── Log Panel ────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        card = tk.Frame(parent, bg=C["bg_card"],
                        highlightbackground=C["border"], highlightthickness=1,
                        padx=2, pady=2)
        card.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=4)

        tk.Label(card, text="📝  Scan History",
                 font=("Segoe UI", 13, "bold"),
                 bg=C["bg_card"], fg="#a78bfa").pack(anchor="w", padx=14, pady=(12, 4))

        sep = tk.Frame(card, bg=C["border"], height=1)
        sep.pack(fill="x", padx=14, pady=4)

        # Scrollable log
        log_container = tk.Frame(card, bg=C["bg_card"])
        log_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_container, bg=C["bg_panel"], fg=C["text"],
                                 font=("Consolas", 10), relief="flat",
                                 insertbackground=C["text"], wrap="word",
                                 state="disabled", padx=10, pady=8)
        scrollbar = ttk.Scrollbar(log_container, orient="vertical",
                                   command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Info panel at bottom
        info_frame = tk.Frame(card, bg=C["bg_panel"], padx=10, pady=8)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(info_frame, text="Supported Countries",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg_panel"], fg="#a78bfa").pack(anchor="w")
        tk.Label(info_frame, text="🇺🇸 🇬🇧",
                 font=("Segoe UI", 18),
                 bg=C["bg_panel"], fg=C["text"]).pack(anchor="w", pady=2)

        ml_status = "Image ML Active ✓" if self.detector.image_ml_available else "No model (run train_image_model.py)"
        tk.Label(info_frame,
                 text=f"US + UK  ·  {ml_status}  ·  Upload Detection",
                 font=("Segoe UI", 8),
                 bg=C["bg_panel"], fg=C["text_dim"]).pack(anchor="w")

    # ═══════════════════════════════════════════════════════════════
    #  FOOTER
    # ═══════════════════════════════════════════════════════════════
    def _build_footer(self):
        footer = tk.Frame(self.win, bg=C["bg"], height=28)
        footer.pack(fill="x", padx=20, pady=(2, 8))

        self.lbl_status_bar = tk.Label(footer, text="Ready — upload an image to begin",
                                        font=("Segoe UI", 9),
                                        bg=C["bg"], fg=C["text_dim"])
        self.lbl_status_bar.pack(side="left")

        ml_tag = "Image ML" if self.detector.image_ml_available else "No Model"
        tk.Label(footer,
                 text=f"Other Country Detection v1.0  ·  {ml_tag}  ·  Upload Mode",
                 font=("Segoe UI", 9),
                 bg=C["bg"], fg=C["text_dim"]).pack(side="right")

    # ═══════════════════════════════════════════════════════════════
    #  UPLOAD & ANALYZE
    # ═══════════════════════════════════════════════════════════════
    def _upload_image(self):
        """Open file dialog to select and analyze a currency image."""
        file_path = filedialog.askopenfilename(
            parent=self.win,
            title="Select Currency Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        self._status(f"Analyzing: {os.path.basename(file_path)} …")
        self.img_status.configure(text="● ANALYZING", fg=C["warning"])

        # Show image preview
        self._show_image_preview(file_path)

        # Analyze in background thread
        import threading
        threading.Thread(
            target=self._analyze_image, args=(file_path,), daemon=True
        ).start()

    def _show_image_preview(self, file_path):
        """Display uploaded image on the canvas."""
        try:
            img = Image.open(file_path)
            # Fit to canvas
            self.img_canvas.update_idletasks()
            cw = max(self.img_canvas.winfo_width(), 300)
            ch = max(self.img_canvas.winfo_height(), 300)

            # Scale maintaining aspect ratio
            img_ratio = img.width / img.height
            canvas_ratio = cw / ch
            if img_ratio > canvas_ratio:
                new_w = cw - 20
                new_h = int(new_w / img_ratio)
            else:
                new_h = ch - 20
                new_w = int(new_h * img_ratio)

            img = img.resize((new_w, new_h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.img_canvas.delete("all")
            self.img_canvas.create_image(cw // 2, ch // 2,
                                          anchor="center", image=imgtk)
            self.img_canvas._imgtk = imgtk  # prevent GC

            # Purple scanning border
            self.img_canvas.create_rectangle(
                4, 4, cw - 4, ch - 4,
                outline="#7c3aed", width=3)
        except Exception as e:
            print(f"[Upload] Preview error: {e}")

    def _analyze_image(self, file_path):
        """Run detection in background thread."""
        try:
            result = self.detector.analyze_uploaded_image(file_path)
            self.current_result = result

            # Update GUI on main thread
            self.win.after(0, lambda: self._update_result(result))

            # Voice feedback
            if result["currency_code"] != "UNKNOWN":
                self.voice.announce_result(result)
        except Exception as e:
            print(f"[Upload] Analysis error: {e}")
            self.win.after(0, lambda: self._status(f"Error: {e}"))

    def _update_result(self, result):
        """Update the result panel with detection results."""
        r = result

        # Flag
        self.lbl_flag.configure(text=r["flag"])

        # Country
        self.lbl_country.configure(text=f"{r['country']}  ({r['currency_code']})")

        # Denomination / currency name
        if r["denomination"]:
            self.lbl_denom.configure(text=f"{r['symbol']}{r['denomination']}")
        else:
            self.lbl_denom.configure(text=r["currency_name"])

        # Status badge
        status = r["status"]
        if status == "REAL":
            bg, label = C["real"], "✓  GENUINE"
        elif status == "SUSPICIOUS":
            bg, label = C["suspicious"], "⚠  SUSPICIOUS"
        elif status == "FAKE":
            bg, label = C["fake"], "✗  FAKE"
        else:
            bg, label = C["unknown"], "—  " + status
        self.lbl_status.configure(text=f"  {label}  ", bg=bg)

        # Image status indicator
        if status == "REAL":
            self.img_status.configure(text="● GENUINE", fg=C["real"])
        elif status == "FAKE":
            self.img_status.configure(text="● FAKE", fg=C["fake"])
        elif status == "SUSPICIOUS":
            self.img_status.configure(text="● SUSPICIOUS", fg=C["suspicious"])
        else:
            self.img_status.configure(text="● ANALYZED", fg=C["text_dim"])

        # Details
        self.lbl_details.configure(text=r["details"])

        # Score bar
        self._draw_bar(self.score_canvas, r["score"], 10,
                       C["real"] if status == "REAL" else
                       C["suspicious"] if status == "SUSPICIOUS" else C["fake"])
        self.lbl_score_val.configure(text=f"{r['score']} / 10")

        # Confidence bar
        self._draw_bar(self.conf_canvas, r["confidence"], 100, "#7c3aed")
        self.lbl_conf_val.configure(text=f"{r['confidence']}%")

        # Log entry
        self._add_log_entry(r)

        self._status(f"Detection complete — {r['country']} {r['status']}")

    def _draw_bar(self, canvas, value, max_val, colour):
        canvas.delete("all")
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 5:
            return
        fill_w = max(0, int((value / max(max_val, 1)) * w))
        canvas.create_rectangle(0, 0, fill_w, h, fill=colour, outline="")

    def _add_log_entry(self, result):
        from datetime import datetime
        sig = f"{result['currency_code']}_{result['denomination']}_{result['status']}"
        if self.scan_log and self.scan_log[-1] == sig:
            return
        self.scan_log.append(sig)

        ts = datetime.now().strftime("%H:%M:%S")
        status_icon = "✓" if result["status"] == "REAL" else \
                      "⚠" if result["status"] == "SUSPICIOUS" else "✗"

        entry = (f"[{ts}] {status_icon} {result['flag']} {result['currency_code']}"
                 f" {result['currency_name']}"
                 f"  →  {result['status']}\n")

        self.log_text.configure(state="normal")
        self.log_text.insert("end", entry)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reset(self):
        self.current_result = None
        self.scan_log.clear()

        self.lbl_flag.configure(text="❓")
        self.lbl_country.configure(text="No Detection Yet")
        self.lbl_denom.configure(text="—")
        self.lbl_status.configure(text=" NO DATA ", bg=C["unknown"])
        self.lbl_details.configure(text="Upload a currency image to\nbegin detection.")
        self.score_canvas.delete("all")
        self.conf_canvas.delete("all")
        self.lbl_score_val.configure(text="0 / 10")
        self.lbl_conf_val.configure(text="0%")
        self.img_status.configure(text="● NO IMAGE", fg=C["text_dim"])

        self.img_canvas.delete("all")
        self.img_canvas.create_text(
            300, 200, text="Click 📁 Upload Image to select\na US or UK currency note image",
            fill=C["text_dim"], font=("Segoe UI", 14), tags="placeholder",
            justify="center")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._status("Detection reset")

    def _status(self, msg):
        try:
            self.lbl_status_bar.configure(text=msg)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyApp(root)
    root.mainloop()
