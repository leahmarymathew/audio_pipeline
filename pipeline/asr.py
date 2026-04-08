import os

import librosa
import numpy as np
import whisper

# Use a smaller default model for faster local processing.
_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
_MODEL = whisper.load_model(_MODEL_NAME)


def compute_asr(audio, sr):
    """Return ASR text and a lightweight confidence proxy."""
    if audio is None or len(audio) == 0:
        return "", 0.0

    # Whisper expects 16kHz mono audio.
    audio_16k = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)

    # Keep inference bounded for responsiveness.
    max_seconds = int(os.getenv("WHISPER_MAX_SECONDS", "20"))
    max_len = 16000 * max_seconds
    if len(audio_16k) > max_len:
        audio_16k = audio_16k[:max_len]

    try:
        result = _MODEL.transcribe(audio_16k, fp16=False, temperature=0.0, best_of=1, beam_size=1)
        text = (result.get("text") or "").strip()
    except Exception as exc:
        print(f"ASR failed: {exc}")
        return "", 0.0

    # Simple confidence proxy from text length.
    confidence = 0.0 if not text else min(len(text) / 50.0, 1.0)
    return text, confidence
