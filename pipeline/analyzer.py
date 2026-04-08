import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_results(df):

    if df.empty:
        print("\n===== OVERALL STATS =====")
        print("No samples were processed.")
        return pd.DataFrame()

    print("\n===== OVERALL STATS =====")

    total = len(df)
    accepted = len(df[df["decision"] == "KEEP"])
    rejected = len(df[df["decision"] == "REJECT"])

    print(f"Total samples: {total}")
    print(f"Accepted: {accepted} ({accepted/total:.2f})")
    print(f"Rejected: {rejected} ({rejected/total:.2f})")

    print("\n===== PER LANGUAGE STATS =====")

    lang_stats = df.groupby("language").agg({
        "score": "mean",
        "snr": "mean",
        "dnsmos": "mean",
        "dnsmos_sig": "mean",
        "dnsmos_bak": "mean",
        
        "decision": lambda x: (x == "KEEP").mean()
    }).rename(columns={
        "decision": "acceptance_rate"
    })

    print(lang_stats)
    plt.figure(figsize=(8, 5))
    df.boxplot(column="score", by="language")
    plt.suptitle("")
    plt.title("Score by Language")
    plt.xlabel("Language")
    plt.ylabel("Score")
    plt.tight_layout()
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/language_score.png")
    print("\n===== QUALITY DISTRIBUTION =====")
    if "quality" in df.columns:
        quality_series = df["quality"]
    elif "score" in df.columns:
        quality_series = pd.cut(
            df["score"],
            bins=[-float("inf"), 0.6, 0.8, float("inf")],
            labels=["low", "medium", "high"],
        )
    else:
        quality_series = pd.Series(["unknown"] * len(df), index=df.index)

    print(quality_series.value_counts())
    print("\n===== REJECTION REASONS =====")
    print(df[df["decision"] == "REJECT"]["reason"].value_counts())
    return lang_stats

    print("\n===== LANGUAGE MISMATCH =====")
    print((df["language"] != df["lang_pred"]).sum())