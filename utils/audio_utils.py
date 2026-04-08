import io

import numpy as np
import librosa
import soundfile as sf


def read_sample_audio(audio_data: dict):
    audio_bytes = audio_data.get("bytes")
    audio_path = audio_data.get("path")

    if audio_bytes is not None:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    elif audio_path:
        audio, sr = sf.read(audio_path, dtype="float32")
    else:
        raise ValueError("Sample does not contain audio bytes or path")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr


def to_model_audio(audio: np.ndarray, sr: int, target_sr: int = 16000, target_len: int = 144160):
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if len(audio_16k) < target_len:
        audio_16k = np.pad(audio_16k, (0, target_len - len(audio_16k)))
    else:
        audio_16k = audio_16k[:target_len]
    return audio_16k.astype("float32")
