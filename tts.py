import sys
import io
import wave
import sounddevice as sd
import numpy as np

# ── Paths ────────────────────────────────────────────────
KOKORO_MODEL  = "/home/ilyes/agents/kokoro-v0_19.onnx"
KOKORO_VOICES = "/home/ilyes/agents/voices-v1.0.bin"
PIPER_MODELS  = "/home/ilyes/agents/piper_models"

# ── Piper model files per language ───────────────────────
PIPER_LANG_MODELS = {
    "es": "es_ES-sharvard-medium.onnx",
    "fr": "fr_FR-siwis-low.onnx",
    "it": "it_IT-riccardo-x_low.onnx",
    "zh": "zh_CN-huayan-medium.onnx",
}

# ── Kokoro voice maps ────────────────────────────────────
KOKORO_VOICES_MAP = {
    "onyx":     "am_onyx",
    "adam":     "am_adam",
    "eric":     "am_eric",
    "fenrir":   "am_fenrir",
    "liam":     "am_liam",
    "michael":  "am_michael",
    "echo":     "am_echo",
    "puck":     "am_puck",
    "george":   "bm_george",
    "lewis":    "bm_lewis",
    "daniel":   "bm_daniel",
    "fable":    "bm_fable",
    "alloy":    "af_alloy",
    "bella":    "af_bella",
    "heart":    "af_heart",
    "jessica":  "af_jessica",
    "nova":     "af_nova",
    "sarah":    "af_sarah",
    "sky":      "af_sky",
    "alice":    "bf_alice",
    "emma":     "bf_emma",
    "isabella": "bf_isabella",
    "lily":     "bf_lily",
    "siwis":    "ff_siwis",      # kokoro french (fallback)
    "dora":     "ef_dora",       # kokoro spanish (fallback)
    "alex":     "em_alex",       # kokoro spanish (fallback)
    "sara":     "if_sara",       # kokoro italian (fallback)
    "nicola":   "im_nicola",     # kokoro italian (fallback)
    "kumo":     "jm_kumo",       # kokoro japanese
}

# ── Language display config ──────────────────────────────
LANGUAGES = [
    ("en (en-us)", "en",  "fenrir,adam,eric,liam,michael,echo,puck,onyx,alloy,bella,heart,jessica,nova,sarah,sky", "kokoro"),
    ("en-gb",      "en-gb","george,lewis,daniel,fable,alice,emma,isabella,lily", "kokoro"),
    ("fr",         "fr",  "siwis (Piper — better quality)", "piper"),
    ("es",         "es",  "mls (Piper — better quality)", "piper"),
    ("it",         "it",  "riccardo (Piper — better quality)", "piper"),
    ("zh",         "zh",  "huayan (Piper — better quality)", "piper"),
    ("ja",         "ja",  "kumo (Kokoro)", "kokoro"),
]

HELP = """
Usage: tts [OPTIONS] TEXT

Options:
  --voice   Kokoro voice name (for English, default: fenrir)
  --lang    Language code (default: en)
  --speed   Speed multiplier (default: 1.1)
  --pick    Interactive language + voice picker
  -h        Show this help

Language codes: en, en-gb, fr, es, it, zh, ja

Engine routing:
  English (en, en-gb) → Kokoro  (best English quality)
  French, Spanish, Italian, Chinese → Piper (better non-English)
  Japanese → Kokoro (no Piper model available)

Examples:
  tts hello world
  tts --voice george good morning
  tts --lang fr bonjour comment ca va
  tts --lang es hola como estas
  tts --pick your text here
"""


def play_wav_bytes(wav_bytes):
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, rate)
    sd.wait()


def speak_piper(text, lang, speed=1.0):
    import os
    model_file = PIPER_LANG_MODELS.get(lang)
    if not model_file:
        print(f"[TTS] No Piper model for '{lang}', falling back to Kokoro")
        return False
    model_path = os.path.join(PIPER_MODELS, model_file)
    if not os.path.exists(model_path):
        print(f"[TTS] Piper model not found: {model_path}")
        return False
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(model_path)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            voice.synthesize_wav(text, wf)
        play_wav_bytes(buf.getvalue())
        return True
    except Exception as e:
        print(f"[TTS] Piper error: {e}, falling back to Kokoro")
        return False


def speak_kokoro(text, voice_id, lang_code, speed=1.1):
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    samples, sample_rate = kokoro.create(text, voice=voice_id, speed=speed, lang=lang_code)
    sd.play(samples, sample_rate)
    sd.wait()


def get_engine(lang):
    """Returns 'piper' or 'kokoro' for a given language code."""
    for _, code, _, engine in LANGUAGES:
        if code == lang:
            return engine
    return "kokoro"


def pick_lang_interactive():
    print("\nAvailable languages:")
    for i, (label, code, voices, engine) in enumerate(LANGUAGES, 1):
        print(f"  {i}. {label} — {voices}")
    choice = input("\nLanguage (name or number, default: en): ").strip().lower()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(LANGUAGES):
            return LANGUAGES[idx][1]
    for label, code, _, _ in LANGUAGES:
        if choice in label.lower() or choice == code:
            return code
    return "en"


def pick_voice_interactive(lang):
    engine = get_engine(lang)
    if engine == "piper":
        print(f"[TTS] Using Piper engine for '{lang}' — no voice selection needed")
        return None
    # Kokoro voice selection
    if lang in ("en", "en-us"):
        voices = ["fenrir","adam","eric","liam","michael","echo","puck","onyx","alloy","bella","heart","jessica","nova","sarah","sky"]
    elif lang == "en-gb":
        voices = ["george","lewis","daniel","fable","alice","emma","isabella","lily"]
    elif lang == "ja":
        voices = ["kumo"]
    else:
        voices = ["fenrir"]
    print(f"Available voices: {', '.join(voices)}")
    choice = input(f"Voice (default: {voices[0]}): ").strip().lower()
    return KOKORO_VOICES_MAP.get(choice, KOKORO_VOICES_MAP.get(voices[0], voices[0]))


def speak_auto(text, lang, voice_id=None, speed=1.1):
    """Auto-route to Piper or Kokoro based on language."""
    engine = get_engine(lang)

    # Kokoro lang codes
    kokoro_lang_map = {
        "en": "en-us", "en-gb": "en-gb",
        "fr": "fr-fr", "es": "es", "it": "it",
        "ja": "ja", "zh": "cmn", "ko": "ko",
    }

    if engine == "piper" and speak_piper(text, lang, speed):
        return

    # Fallback or English: use Kokoro
    if not voice_id:
        voice_id = "am_fenrir"
    lang_code = kokoro_lang_map.get(lang, "en-us")
    speak_kokoro(text, voice_id, lang_code, speed)


def speak_interactive(text: str):
    """Called by main.py TTS/READ agents — interactive picker then speak."""
    print(f"\n[TTS] Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    lang = pick_lang_interactive()
    voice_id = pick_voice_interactive(lang)
    speed_input = input("Speed (default 1.1): ").strip()
    speed = float(speed_input) if speed_input else 1.1
    speak_auto(text, lang, voice_id, speed)


def parse_args(args):
    if "-h" in args or "--help" in args:
        print(HELP)
        sys.exit(0)
    voice = None
    lang = "en"
    speed = 1.1
    text_parts = []
    interactive = False
    i = 0
    while i < len(args):
        if args[i] == "--voice" and i + 1 < len(args):
            voice = KOKORO_VOICES_MAP.get(args[i+1].lower(), args[i+1].lower())
            i += 2
        elif args[i] == "--lang" and i + 1 < len(args):
            lang = args[i+1].lower()
            i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = float(args[i+1])
            i += 2
        elif args[i] == "--pick":
            interactive = True
            i += 1
        else:
            text_parts.append(args[i])
            i += 1
    return voice, lang, speed, " ".join(text_parts), interactive


def main():
    voice, lang, speed, text, interactive = parse_args(sys.argv[1:])
    if not text:
        text = input("Text to speak: ").strip()
    if interactive:
        lang = pick_lang_interactive()
        voice = pick_voice_interactive(lang)
        speed_input = input("Speed (default 1.1): ").strip()
        speed = float(speed_input) if speed_input else 1.1
    speak_auto(text, lang, voice, speed)


if __name__ == "__main__":
    main()
