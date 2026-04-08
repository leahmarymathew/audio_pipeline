import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path


Path("plots").mkdir(exist_ok=True)

for i, sample in enumerate(rows[:3]):  # take few samples
    # reuse audio from earlier loop or reload if needed
    # here assuming you saved audio_16k temporarily

    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    plt.title(f"Sample {i} - {sample['decision']}")
    plt.tight_layout()

    plt.savefig(f"plots/sample_{i}.png")
    plt.close()