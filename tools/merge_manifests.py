"""
Merge per-dataset manifest.parquet files into a single manifest.parquet.
"""
from pathlib import Path
import pandas as pd

# List of per-dataset manifest paths (edit as needed)
manifest_paths = list(Path("./prepared").rglob("manifest.parquet"))

if not manifest_paths:
    raise FileNotFoundError("No manifest.parquet files found under ./prepared/")

print(f"Found {len(manifest_paths)} manifest files:")
for p in manifest_paths:
    print(" -", p)

# Load and concatenate all manifests
dfs = [pd.read_parquet(p) for p in manifest_paths]
merged_df = pd.concat(dfs, ignore_index=True)

# Write merged manifest
output_path = Path("./prepared/manifest.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_parquet(output_path, index=False)
print(f"Merged manifest written to {output_path} with {len(merged_df)} samples.")
