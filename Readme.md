Approach

The pipeline evaluates each audio sample using multiple metrics and computes a composite quality score.

1. Signal-Level Metrics
SNR (Signal-to-Noise Ratio) – measures noise level
Silence Ratio – computed using frame-based energy
Clipping Detection – detects distortion
Duration Check – removes very short clips
2. Neural Perceptual Metrics
DNSMOS – predicts human-perceived audio quality
SIG (speech quality)
BAK (background noise)
OVRL (overall score)
3. Language Verification
Language Identification using wav2vec2
Ensures audio matches expected language (Hindi)
4. Intelligibility Estimation
Whisper ASR used to compute average log-probability
Low confidence → unintelligible speech
5. Composite Scoring

Final score is computed as:

Weighted combination of:
SNR
Speech ratio
DNSMOS
ASR confidence
Clipping penalty
Filtering Strategy
Hard Rejection Rules
Duration < 1 second
High clipping (>10%)
Language mismatch
DNSMOS < threshold
Soft Filtering
Composite score threshold determines final decision
Output

The pipeline generates a CSV file with:

Audio metadata
Computed metrics
Quality score
Final decision (KEEP / REJECT)
Rejection reasons
Results
Successfully filters noisy and low-quality samples
Provides explainable decisions
Combines multiple metrics for robust evaluation
Key Contributions
Multi-metric audio quality evaluation
Integration of neural perceptual models (DNSMOS)
Language-aware filtering
ASR-based intelligibility scoring
Scalable streaming pipeline
Technologies Used
Python
librosa
PyTorch
ONNX Runtime
Hugging Face Datasets
Whisper
Future Work
Add UTMOS for improved perceptual scoring
Speaker consistency detection
Distributed processing using Ray
Per-language threshold tuning