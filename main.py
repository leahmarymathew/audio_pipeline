import pandas as pd
from pathlib import Path

from pipeline.loader import get_datasets
from pipeline.filter import process_datasets
from pipeline.metrics import compute_metrics
from pipeline.analyzer import analyze_results
from pipeline.visualizer import plot_all


datasets_dict = get_datasets()

rows = process_datasets(datasets_dict, compute_metrics)

df = pd.DataFrame(rows)

if df.empty:
	print(df.head())
	print(f"\nTotal samples processed: {len(df)}")
	print("No rows were produced, so analysis and plots were skipped.")
	raise SystemExit(0)

Path("output").mkdir(exist_ok=True)
df.to_csv("output/results.csv", index=False)


print(df.head())
print(f"\nTotal samples processed: {len(df)}")


lang_stats = analyze_results(df)

# Save stats
lang_stats.to_csv("output/language_stats.csv")

plot_all(df)

