import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)

def save_spectrogram(audio, sr, filename):

    import numpy as np

    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.savefig(f"plots/{filename}")
    plt.close()