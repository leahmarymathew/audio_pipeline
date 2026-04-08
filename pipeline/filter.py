import io
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from pipeline.asr import compute_asr
from pipeline.langid import detect_language
from pipeline.visual_audio import save_spectrogram


def _load_max_samples(default_value=10):
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return default_value

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return int(config.get("settings", {}).get("max_samples", default_value))
    except Exception as exc:
        print(f"Config load warning: {exc}")
        return default_value


def process_datasets(datasets_dict, compute_metrics, max_samples=None):
    if max_samples is None:
        max_samples = _load_max_samples(default_value=10)

    rows = []

    for lang, ds in datasets_dict.items():
        print(f"Processing {lang}...")
        iterator = iter(ds)
        processed = 0
        attempts = 0
        max_attempts = max_samples * 5

        while processed < max_samples and attempts < max_attempts:
            attempts += 1
            try:
                sample = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                print(f"[{lang}] stream read warning: {exc}")
                continue

            try:
                audio_data = sample["audio_filepath"]
                audio_bytes = audio_data.get("bytes")
                audio_path = audio_data.get("path")

                if audio_bytes is not None:
                    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                elif audio_path:
                    audio, sr = sf.read(audio_path, dtype="float32")
                else:
                    continue

                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                # -------- ASR + LangID --------
                text, asr_conf = compute_asr(audio, sr)
                try:
                    lang_pred, lang_conf = detect_language(text)
                except Exception as exc:
                    print(f"[{lang}] language-id warning: {exc}")
                    lang_pred, lang_conf = "unknown", 0.0

                # -------- Metrics --------
                result = compute_metrics(audio, sr, asr_conf, lang_pred, lang_conf, lang)

                # 🔥 -------- SPECTROGRAM SAVE (ADDED HERE) --------
                try:
                    if processed < 2:  # first 2 samples per language
                        if result["decision"] == "KEEP":
                            save_spectrogram(audio, sr, f"good_{lang}_{processed}.png")
                        else:
                            save_spectrogram(audio, sr, f"bad_{lang}_{processed}.png")
                except Exception as exc:
                    print(f"[{lang}] spectrogram warning: {exc}")

                # -------- STORE RESULT --------
                rows.append(
                    {
                        "file_id": processed,
                        "language": lang,
                        "asr_text": text,
                        "asr_confidence": asr_conf,
                        "language_prediction": lang_pred,
                        "language_confidence": lang_conf,
                        **result,
                    }
                )

                processed += 1

            except Exception as exc:
                print(f"[{lang}] sample processing warning: {exc}")

    return rows