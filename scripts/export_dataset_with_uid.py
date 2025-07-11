import pandas as pd
from datasets import load_from_disk

def export_to_csv(dataset_path, split=None, output_csv="dataset_with_uid.csv"):
    # Load the dataset from disk
    ds = load_from_disk(dataset_path)
    # If split is specified and present, select it
    if split is not None and split in ds:
        ds = ds[split]
    # Convert to pandas DataFrame
    df = ds.to_pandas()
    # Ensure 'universal_id' is present
    if 'universal_id' not in df.columns:
        raise ValueError("Column 'universal_id' not found in dataset.")
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    export_to_csv(
        dataset_path="/root/code/verl/dataset/processed_with_uid_hf",
        split="train",
        output_csv="/root/code/verl/dataset/processed_with_uid_hf/dataset_with_uid.csv"
    )