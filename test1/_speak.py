import sys, os
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