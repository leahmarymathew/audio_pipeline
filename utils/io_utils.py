from pathlib import Path

import pandas as pd


def save_results(rows, csv_path: str = "output/results.csv"):
    df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def append_log(message: str, log_path: str = "output/logs.txt"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
