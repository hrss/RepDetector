import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum

# Reusing the logic from loso_eval.py to keep labels consistent
class ExerciseLabel(Enum):
    AIR_SQUAT = "Air Squat"
    BURPEE = "Burpee"
    KB_SWING = "KB swing"
    WALL_BALL = "Wall ball"
    PUSH_UP = "Push-up"
    REST = "REST"
    SIT_UP = "Sit-up"
    WALKING_LUNGE = "Walking lunge"
    BOX_JUMP = "Box jump"
    DOUBLE_UNDER = "Double-under"
    RUN = "Run"
    RUN_ALL_OUT = "Run All Out"

LABEL_MAP = {
    "rest": ExerciseLabel.REST,
    "air squat": ExerciseLabel.AIR_SQUAT,
    "air_squat": ExerciseLabel.AIR_SQUAT,
    "burpee": ExerciseLabel.BURPEE,
    "kb swing": ExerciseLabel.KB_SWING,
    "kb_swing": ExerciseLabel.KB_SWING,
    "kb swing (russian)": ExerciseLabel.KB_SWING,
    "kb swing (american)": ExerciseLabel.KB_SWING,
    "wall ball": ExerciseLabel.WALL_BALL,
    "wall_ball_shot": ExerciseLabel.WALL_BALL,
    "push-up": ExerciseLabel.PUSH_UP,
    "chest_to_wall_hspu": ExerciseLabel.PUSH_UP,
    "sit-up": ExerciseLabel.SIT_UP,
    "sit_up": ExerciseLabel.SIT_UP,
    "walking lunge": ExerciseLabel.WALKING_LUNGE,
    "sandbag_lunges": ExerciseLabel.WALKING_LUNGE,
    "box jump": ExerciseLabel.BOX_JUMP,
    "box_jump": ExerciseLabel.BOX_JUMP,
    "double-under": ExerciseLabel.DOUBLE_UNDER,
    "double_under": ExerciseLabel.DOUBLE_UNDER,
    "run": ExerciseLabel.RUN,
    "run all out": ExerciseLabel.RUN_ALL_OUT,
}

IGNORE_LABELS = ["null", "setup"]

def normalize_label(label_str):
    if not label_str or pd.isna(label_str):
        return None
    ls = label_str.strip().lower()
    if ls in IGNORE_LABELS:
        return None
    
    # Direct match
    if ls in LABEL_MAP:
        return LABEL_MAP[ls].value
    
    # Substring matches for variants
    if "kb swing" in ls:
        return ExerciseLabel.KB_SWING.value
    if "rest" in ls:
        return ExerciseLabel.REST.value
    if "wall ball" in ls:
        return ExerciseLabel.WALL_BALL.value
    if "lunge" in ls:
        return ExerciseLabel.WALKING_LUNGE.value
    if "box jump" in ls:
        return ExerciseLabel.BOX_JUMP.value
    if "double under" in ls or "double-under" in ls:
        return ExerciseLabel.DOUBLE_UNDER.value
    if "push up" in ls or "push-up" in ls:
        return ExerciseLabel.PUSH_UP.value
    if "sit up" in ls or "sit-up" in ls:
        return ExerciseLabel.SIT_UP.value
    if "air squat" in ls:
        return ExerciseLabel.AIR_SQUAT.value
    if "burpee" in ls:
        return ExerciseLabel.BURPEE.value
    if "run all out" in ls:
        return ExerciseLabel.RUN_ALL_OUT.value
    if "run" in ls:
        return ExerciseLabel.RUN.value
        
    return label_str # Keep unknown labels as they are for inventory visibility

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
                    norm_name = normalize_label(name)
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
