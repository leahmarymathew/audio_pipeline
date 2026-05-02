# Audio Quality Filtering Pipeline

An automated, multi-metric pipeline for evaluating and filtering speech audio samples from the [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) dataset. The pipeline streams audio for five Indian languages, computes signal-level and neural perceptual metrics, and produces an explainable KEEP / REJECT decision per sample.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Metrics](#metrics)
5. [Scoring & Filtering Logic](#scoring--filtering-logic)
6. [Configuration](#configuration)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Outputs](#outputs)
10. [Technologies Used](#technologies-used)
11. [Future Work](#future-work)

---

## Overview

Low-quality audio in speech datasets degrades downstream model training. This pipeline provides a scalable, streaming quality-control layer that combines classical signal processing with neural perceptual scoring to automatically identify and reject noisy, silent, clipped, or unintelligible recordings.

Supported languages: **Hindi, Tamil, Bengali, Marathi, Gujarati**

---

## Repository Structure

```
audio_pipeline/
├── main.py                  # Pipeline entry point
├── graph.py                 # Standalone spectrogram viewer (debug utility)
├── sig_bak_ovr.onnx         # Pre-trained DNSMOS ONNX model
├── requirements.txt
├── config/
│   └── config.yaml          # Thresholds, weights, and run settings
├── pipeline/
│   ├── loader.py            # Streams IndicVoices dataset from Hugging Face
│   ├── filter.py            # Main processing loop (audio I/O, ASR, LangID, metrics)
│   ├── metrics.py           # Signal metrics + composite scoring + decision logic
│   ├── dnsmos.py            # DNSMOS inference via ONNX Runtime
│   ├── asr.py               # Whisper-based ASR + confidence estimation
│   ├── langid.py            # Language identification via langid
│   ├── scorer.py            # Standalone metric normalisation helpers
│   ├── analyzer.py          # Per-language statistics and box-plot generation
│   ├── visualizer.py        # Distribution plots (score, SNR, DNSMOS, quality, ASR)
│   └── visual_audio.py      # Per-sample spectrogram saver
└── utils/
    ├── audio_utils.py       # Audio loading and resampling helpers
    └── io_utils.py          # CSV and log file writers
```

---

## Pipeline Architecture

```
IndicVoices (streaming)
        │
        ▼
  pipeline/loader.py      ← streams audio for each language
        │
        ▼
  pipeline/filter.py      ← decodes audio bytes / path → mono float32
        │
        ├──► pipeline/asr.py        → transcript + ASR confidence (Whisper)
        ├──► pipeline/langid.py     → detected language + confidence (langid)
        └──► pipeline/metrics.py    → all signal metrics + DNSMOS + decision
                │
                ├──► pipeline/dnsmos.py     → SIG / BAK / OVRL scores (ONNX)
                └──► pipeline/visual_audio.py → spectrogram PNG per sample
        │
        ▼
  main.py
        ├──► output/results.csv          (per-sample metrics & decisions)
        ├──► output/language_stats.csv   (per-language aggregated stats)
        └──► plots/                      (distribution & diagnostic charts)
```

---

## Metrics

### Signal-Level Metrics

| Metric | Description |
|---|---|
| **Duration** | Length of the audio clip in seconds |
| **WADA-SNR** | Signal-to-noise ratio via the WADA estimator (no clean reference needed) |
| **C50** | Clarity index — ratio of early (< 50 ms) to late energy in dB |
| **Silence Ratio** | Fraction of 20 ms frames whose RMS energy is below threshold |
| **Clipping Ratio** | Fraction of samples whose absolute amplitude exceeds 0.99 |

### Neural Perceptual Metrics (DNSMOS)

Computed using a pre-trained ONNX model (`sig_bak_ovr.onnx`) on 16 kHz audio padded/trimmed to 9 seconds (144,160 samples).

| Score | Meaning |
|---|---|
| **SIG** | Predicted speech signal quality (1–5) |
| **BAK** | Predicted background noise intrusiveness (1–5) |
| **OVRL** | Predicted overall quality (1–5) — used as `dnsmos` in scoring |

### ASR Intelligibility (Whisper)

OpenAI Whisper (`tiny` model by default, overridable via `WHISPER_MODEL` env var) transcribes the audio. Confidence is approximated as `min(len(transcript) / 50, 1.0)`.

### Language Identification (langid)

The `langid` library classifies the transcribed text. A mismatch with the expected language applies a score penalty of −0.10.

---

## Scoring & Filtering Logic

### Composite Score

```
score = 0.35 × dnsmos_norm
      + 0.20 × speech_ratio
      + 0.20 × snr_norm
      + 0.15 × (1 − clipping)
      + 0.10 × asr_confidence
```

Where normalised values are clamped to [0, 1]:
- `dnsmos_norm = min(dnsmos / 5, 1)`
- `snr_norm = min(snr / 20, 1)`
- `speech_ratio = 1 − silence_ratio`

### Hard Rejection Rules (applied first)

| Condition | Threshold |
|---|---|
| DNSMOS (OVRL) | < 2.5 |
| ASR confidence | < 0.2 |
| Silence ratio | > 0.5 |

### Soft Rejection Rule

If the sample passed all hard filters, it is rejected if `score < 0.55`.

### Language Mismatch

A mismatch between detected and expected language does **not** immediately reject the sample; instead, the composite score is reduced by 0.10 (which may trigger the soft-filter threshold).

---

## Configuration

All thresholds, scoring weights, and run settings are controlled by `config/config.yaml`:

```yaml
settings:
  max_samples: 10          # Samples to evaluate per language
  language: hi             # Primary target language code
  target_sr: 16000         # Target sample rate for models (Hz)
  dnsmos_input_len: 144160 # Fixed input length for DNSMOS model (~9 s at 16 kHz)

thresholds:
  min_duration_sec: 1.0
  max_clipping_ratio: 0.1
  max_silence_ratio: 0.6
  min_snr_db: 5.0
  min_dnsmos: 2.5
  min_whisper_conf: -1.5
  min_score: 0.4

weights:
  snr: 0.2
  speech_ratio: 0.2
  dnsmos: 0.3
  whisper: 0.2
  unclipped: 0.1
```

> **Note:** The weights in `config.yaml` are reference values. The active weights used in `pipeline/metrics.py` are hard-coded and slightly different (DNSMOS 0.35, clipping 0.15). Centralising these into the config loader is a planned improvement.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/leahmarymathew/audio_pipeline.git
cd audio_pipeline

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`):

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data manipulation |
| `soundfile`, `librosa` | Audio I/O and processing |
| `onnxruntime` | DNSMOS model inference |
| `datasets` | Streaming IndicVoices from Hugging Face |
| `openai-whisper` | ASR transcription and confidence |
| `transformers`, `torch` | Deep-learning backend |
| `langid` | Language identification |
| `matplotlib`, `seaborn` | Plotting |
| `pyyaml` | Config file parsing |

---

## Usage

```bash
python main.py
```

This will:
1. Stream `max_samples` audio samples per language from IndicVoices.
2. Run the full metrics pipeline on each sample.
3. Write results to `output/results.csv` and `output/language_stats.csv`.
4. Save diagnostic plots to `plots/`.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `tiny` | Whisper model size (`tiny`, `base`, `small`, …) |
| `WHISPER_MAX_SECONDS` | `20` | Maximum audio length (seconds) sent to Whisper |

---

## Outputs

### `output/results.csv`

One row per processed audio sample with the following columns:

| Column | Description |
|---|---|
| `file_id` | Sample index within its language |
| `language` | Source language |
| `asr_text` | Whisper transcript |
| `asr_confidence` | Confidence proxy (0–1) |
| `language_prediction` | Language detected by langid |
| `language_confidence` | langid confidence score |
| `duration` | Clip duration in seconds |
| `snr` | WADA-SNR estimate (dB) |
| `c50` | Clarity index (dB) |
| `silence` | Silence ratio (0–1) |
| `clipping` | Clipping ratio (0–1) |
| `dnsmos` | DNSMOS OVRL score (1–5) |
| `dnsmos_sig` | DNSMOS SIG score (1–5) |
| `dnsmos_bak` | DNSMOS BAK score (1–5) |
| `speech_ratio` | Active speech ratio (0–1) |
| `score` | Composite quality score (0–1) |
| `decision` | `KEEP` or `REJECT` |
| `reason` | Comma-separated rejection reasons (if `REJECT`) |

### `output/language_stats.csv`

Per-language aggregated statistics: mean score, SNR, DNSMOS (SIG/BAK/OVRL), and acceptance rate.

### `plots/`

| File | Description |
|---|---|
| `score_distribution.png` | Histogram + KDE of composite scores |
| `snr_distribution.png` | Histogram + KDE of SNR values |
| `dnsmos_distribution.png` | Histogram + KDE of DNSMOS OVRL values |
| `asr_conf.png` | Histogram + KDE of ASR confidence |
| `keep_vs_reject.png` | Pie chart of KEEP vs REJECT decisions |
| `quality_distribution.png` | Bar chart of low / medium / high quality bins |
| `language_score.png` | Box plot of composite score by language |
| `good_<lang>_<n>.png` | Spectrogram of a kept sample |
| `bad_<lang>_<n>.png` | Spectrogram of a rejected sample |

---

## Technologies Used

| Technology | Role |
|---|---|
| Python 3.x | Core language |
| librosa | Audio loading, resampling, spectrograms |
| soundfile | Low-level audio I/O |
| ONNX Runtime | DNSMOS neural model inference |
| OpenAI Whisper | ASR and intelligibility estimation |
| langid | Text-based language identification |
| Hugging Face `datasets` | Streaming IndicVoices dataset |
| PyTorch / Transformers | Deep-learning model support |
| matplotlib / seaborn | Visualisation |
| PyYAML | Configuration management |

---

## Future Work

- Centralise all scoring weights into `config/config.yaml` to avoid hard-coded values in `metrics.py`
- Add **UTMOS** for improved neural perceptual scoring
- Implement **speaker consistency detection** to filter multi-speaker clips
- Support **distributed processing** via Ray for large-scale dataset runs
- Enable **per-language threshold tuning** based on empirical distributions
- Add a **CLI interface** (e.g., `argparse`) for easier configuration override at runtime
