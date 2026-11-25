"""
Voice Assistant Module - Text-to-Speech via subprocess
Speaks detection results without blocking Tkinter.
Uses temp file to pass text (avoids command-line char issues).
"""

import subprocess
import sys
import threading
import time
import os
import tempfile


_SPEAK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_speak.py")


def _ensure_speak_script():
    """Create the helper speak script if it doesn't exist."""
    code = r'''import sys, os
try:
    text_file = sys.argv[1]
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    os.remove(text_file)
    if not text:
        sys.exit(0)
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if voices and len(voices) > 1:
        engine.setProperty("voice", voices[1].id)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
except Exception as e:
    print(f"Speech error: {e}", file=sys.stderr)
'''
    with open(_SPEAK_SCRIPT, "w", encoding="utf-8") as f:
        f.write(code.strip())


class VoiceAssistant:
    """
    Voice assistant using subprocess for TTS.
    Completely avoids COM/threading issues on Windows.
    """

    def __init__(self):
        self._last_signature = ""     # simple signature, not full text
        self._last_time = 0
        self._cooldown = 10.0         # seconds between speeches
        self._enabled = True
        self._speaking = False        # flag: is speech in progress?
        self._lock = threading.Lock()
        _ensure_speak_script()
        print("[VoiceAssistant] Ready (subprocess mode)")

    def speak(self, text, force=False):
        """Speak text via subprocess. Non-blocking."""
        if not self._enabled or not text or not text.strip():
            return

        # Don't queue if already speaking
        if self._speaking and not force:
            return

        now = time.time()
        # Time-based cooldown only (ignore text content)
        if not force and (now - self._last_time) < self._cooldown:
            return

        self._last_time = now

        # Run in background thread
        t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
        t.start()

    def _do_speak(self, text):
        """Spawn subprocess to speak."""
        if not self._lock.acquire(blocking=False):
            return  # another speech in progress, skip

        self._speaking = True
        try:
            # Write text to temp file (avoids cmd-line escaping issues)
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="voice_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)

            python = sys.executable
            flags = 0
            if os.name == 'nt':
                flags = subprocess.CREATE_NO_WINDOW

            proc = subprocess.Popen(
                [python, _SPEAK_SCRIPT, tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags
            )
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.terminate()
        except Exception as e:
            print(f"[Voice] Error: {e}")
            # Clean up temp file if still exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        finally:
            self._speaking = False
            self._lock.release()

    def announce_result(self, result):
        """Announce detection result with full explanation."""
        if result["currency_code"] == "UNKNOWN":
            return

        # Use a simple signature to avoid repeating exact same detection
        sig = f"{result['currency_code']}_{result.get('denomination', '')}_{result['status']}"

        # Skip if same detection and recently spoken
        now = time.time()
        if sig == self._last_signature and (now - self._last_time) < self._cooldown:
            return

        self._last_signature = sig

        voice_reason = result.get("voice_reason", "")
        if voice_reason:
            self.speak(voice_reason, force=True)
        else:
            parts = []
            status = result["status"]
            if status == "REAL":
                parts.append("Detected genuine")
            elif status == "SUSPICIOUS":
                parts.append("Suspicious note detected.")
            else:
                parts.append("Warning: possibly fake note.")
            parts.append(result["currency_name"])
            if result["denomination"]:
                parts.append(f"denomination {result['denomination']}")
            parts.append(f"from {result['country']}")
            self.speak(" ".join(parts), force=True)

    def toggle(self):
        """Toggle voice on/off."""
        self._enabled = not self._enabled
        return self._enabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)

    def shutdown(self):
        """Cleanup."""
        pass
