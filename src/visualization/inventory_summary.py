import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum

from src.core.exercises import canonicalize_label

def scan_inventory(data_dir):
    meta_files = glob.glob(os.path.join(data_dir, "**", "*.meta.json"), recursive=True)
    # Also check for plain .json files if they are not .meta.json
    all_json = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    for f in all_json:
        if not f.endswith(".meta.json") and ".meta" not in f and f not in meta_files:
            meta_files.append(f)

    exercise_stats = {} # {exercise_name: {'seconds': 0.0, 'sessions': set()}}

    for meta_path in meta_files:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            session_id = meta.get('session_id', os.path.basename(meta_path))
            
            segments = []
            if "segments" in meta:
                segments = meta["segments"]
            elif "exercises" in meta:
                segments = meta["exercises"]
            elif "workout" in meta and isinstance(meta["workout"], dict):
                w = meta["workout"]
                if "segments" in w:
                    segments = w["segments"]
                elif "exercises" in w:
                    segments = w["exercises"]
                elif "roundResults" in w:
                    for r in w["roundResults"]:
                        segments.extend(r.get("exerciseResults", []))
            elif "roundResults" in meta:
                 for r in meta["roundResults"]:
                    segments.extend(r.get("exerciseResults", []))

            current_time = 0.0
            for seg in segments:
                name = seg.get('name') or seg.get('canonicalName')
                start = seg.get('start') or seg.get('startTime')
                end = seg.get('end') or seg.get('endTime')
                
                if start is None or end is None:
                    duration = 0.0
                    if "endConditionValues" in seg and seg["endConditionValues"]:
                        duration = float(seg["endConditionValues"][0])
                    elif "duration" in seg:
                        duration = float(seg["duration"])
                    
                    start = current_time
                    end = current_time + duration
                
                current_time = end
                
                if name:
                    norm_name = canonicalize_label(name)
                    if norm_name is None:
                        continue
                        
                    duration = end - start
                    if norm_name not in exercise_stats:
                        exercise_stats[norm_name] = {'seconds': 0.0, 'sessions': set()}
                    
                    exercise_stats[norm_name]['seconds'] += duration
                    exercise_stats[norm_name]['sessions'].add(session_id)
        except Exception as e:
            print(f"Error processing {meta_path}: {e}")

    return exercise_stats

def plot_inventory(stats, output_file="exercise_inventory.png"):
    if not stats:
        print("No data found to plot.")
        return

    df = pd.DataFrame([
        {'Exercise': k, 'Minutes': v['seconds'] / 60.0, 'Sessions': len(v['sessions'])}
        for k, v in stats.items()
    ])
    
    df = df.sort_values(by='Minutes', ascending=True)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(df['Exercise'], df['Minutes'], color='skyblue')
    
    # Add labels to bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                 f"{df['Minutes'].iloc[i]:.1f}m ({df['Sessions'].iloc[i]} sess)", 
                 va='center')

    plt.xlabel('Total Collected Data (Minutes)')
    plt.title('Exercise Data Inventory Summary')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

    # Also print a nice table
    print("\n--- Exercise Inventory Summary ---")
    print(f"{'Exercise':<25} | {'Minutes':<10} | {'Sessions':<10}")
    print("-" * 50)
    for _, row in df.sort_values(by='Minutes', ascending=False).iterrows():
        print(f"{row['Exercise']:<25} | {row['Minutes']:>8.1f}m | {row['Sessions']:>8}")

if __name__ == "__main__":
    data_directory = "data/processed/apple"
    stats = scan_inventory(data_directory)
    plot_inventory(stats)
