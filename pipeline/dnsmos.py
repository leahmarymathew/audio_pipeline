import numpy as np
import librosa
import onnxruntime as ort

session = ort.InferenceSession("sig_bak_ovr.onnx")

def compute_dnsmos(audio, sr):

    # resample to 16k
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # pad / trim
    target_len = 144160
    if len(audio_16k) < target_len:
        audio_16k = np.pad(audio_16k, (0, target_len - len(audio_16k)))
    else:
        audio_16k = audio_16k[:target_len]

    audio_input = audio_16k.astype(np.float32)[None, :]

    sig, bak, ovr = session.run(None, {"input_1": audio_input})[0][0]

    return sig, bak, ovr