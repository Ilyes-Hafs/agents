import sys
import sounddevice as sd
from kokoro_onnx import Kokoro

VOICES = {
    "onyx":    "am_onyx",
    "adam":    "am_adam",
    "eric":    "am_eric",
    "fenrir":  "am_fenrir",
    "michael": "am_michael",
    "george":  "bm_george",
    "lewis":   "bm_lewis",
    "daniel":  "bm_daniel",
}

LANGS = {
    "en":    "en-us",
    "en-gb": "en-gb",
    "fr":    "fr-fr",
    "ja":    "ja",
    "ko":    "ko",
    "zh":    "cmn",
}

HELP = """
Usage: tts [OPTIONS] TEXT

Options:
  --voice   Voice to use (default: george)
            Choices: onyx, adam, eric, fenrir, michael, george, lewis, daniel
  --lang    Language (default: en)
            Choices: en, en-gb, fr, ja, ko, zh
  --speed   Speed multiplier (default: 1.2)
  -h        Show this help message

Examples:
  tts hello world
  tts --voice lewis hello there
  tts --lang fr bonjour
  tts --speed 1.4 good morning sir
"""

def parse_args(args):
    if "-h" in args or "--help" in args:
        print(HELP)
        sys.exit(0)

    voice = "am_fenrir"
    lang = "en-gb"
    speed = 1.2
    text_parts = []

    i = 0
    while i < len(args):
        if args[i] == "--voice" and i + 1 < len(args):
            v = args[i+1].lower()
            voice = VOICES.get(v, v)
            i += 2
        elif args[i] == "--lang" and i + 1 < len(args):
            l = args[i+1].lower()
            lang = LANGS.get(l, l)
            i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = float(args[i+1])
            i += 2
        else:
            text_parts.append(args[i])
            i += 1

    return voice, lang, speed, " ".join(text_parts)

voice, lang, speed, text = parse_args(sys.argv[1:])

if not text:
    text = input("Text to speak: ")

kokoro = Kokoro("kokoro-v0_19.onnx", "voices-v1.0.bin")
samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
sd.play(samples, sample_rate)
sd.wait()
