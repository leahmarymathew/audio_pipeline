import pandas as pd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

Path("plots").mkdir(exist_ok=True)

def plot_all(df):
    # Compute quality from score if not present
    if "quality" not in df.columns:
        df = df.copy()
        df["quality"] = pd.cut(
            df["score"],
            bins=[-float("inf"), 0.6, 0.8, float("inf")],
            labels=["low", "medium", "high"],
        )

    # -------- Score Distribution --------
    plt.figure()
    sns.histplot(df["score"], bins=30, kde=True)
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.savefig("plots/score_distribution.png")
    plt.close()

    # -------- SNR Distribution --------
    plt.figure()
    sns.histplot(df["snr"], bins=30, kde=True)
    plt.title("SNR Distribution")
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.savefig("plots/snr_distribution.png")
    plt.close()

    # -------- DNSMOS Distribution --------
    plt.figure()
    sns.histplot(df["dnsmos"], bins=30, kde=True)
    plt.title("DNSMOS Distribution")
    plt.xlabel("DNSMOS")
    plt.ylabel("Count")
    plt.savefig("plots/dnsmos_distribution.png")
    plt.close()

    # -------- KEEP vs REJECT --------
    plt.figure()
    df["decision"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("KEEP vs REJECT")
    plt.ylabel("")
    plt.savefig("plots/keep_vs_reject.png")
    plt.close()

    # -------- Quality Distribution --------
    plt.figure()
    df["quality"].value_counts().plot(kind="bar")
    plt.title("Quality Distribution")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.savefig("plots/quality_distribution.png")
    plt.close()

    # -------- ASR Confidence Distribution --------
    plt.figure()
    sns.histplot(df["asr_conf"], bins=30, kde=True)
    plt.title("ASR Confidence Distribution")
    plt.xlabel("ASR Confidence")
    plt.ylabel("Count")
    plt.savefig("plots/asr_conf.png")
    plt.close()