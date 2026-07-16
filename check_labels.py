
import os
import glob
import pandas as pd
import json

def check_labels(data_dir):
    parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    all_labels = []
    
    config = {'lowpass_cutoff': 3.0, 'sample_rate': 20} # dummy config for load_raw_section_data if I were to use it
    
    # But I can just look at the metadata or the parquet if labels are already attached.
    # Actually load_raw_section_data attaches labels.
    
    from src.training.data_loader import load_raw_section_data
    
    for p in parquet_files:
        prefix = p.replace(".parquet", "")
        df, meta = load_raw_section_data(prefix, config)
        if df is not None:
            all_labels.extend(df['label'].unique())
            print(f"File: {os.path.basename(p)}")
            print(df['label'].value_counts())
            print("-" * 20)

if __name__ == "__main__":
    check_labels("data/processed/apple")
