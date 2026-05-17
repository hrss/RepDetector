import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_ribbons(ax, y_pos, labels, times, title, color_map):
    if len(times) < 2: return

    # Convert Pandas Series to Numpy arrays to prevent indexing errors
    times = np.array(times)
    labels = np.array(labels)

    step = times[1] - times[0]
    for t, lab in zip(times, labels):
        rect = mpatches.Rectangle((t - step / 2, y_pos), step, 0.8, color=color_map.get(lab, 'gray'))
        ax.add_patch(rect)

    # Now times[-1] works perfectly to get the last element
    offset = (times[-1] - times[0]) * 0.02
    ax.text(times[0] - offset, y_pos + 0.4, title, va='center', ha='right', fontweight='bold')

def plot_classification_results(res_df, metadata, label_encoder, save_name):
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # 1. Force all classes to standard Python strings
    all_seen_classes = set(res_df['truth']) | set(res_df['pred_raw']) | set(res_df['pred_smooth'])
    unique_classes = sorted(list(all_seen_classes))

    # 2. Map colors properly using zip
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_classes))]
    color_map = {cls: col for cls, col in zip(unique_classes, colors)}
    color_map['Rest'] = '#e0e0e0'  # Override 'Rest' to be gray

    # Top Plot: The Signal Context
    ax1.plot(res_df['time'], res_df['acc_z'], color='black', alpha=0.7, linewidth=1, label='Accel Z (Filtered)')
    section_id = metadata.get("sectionId", "Unknown")[:8]
    ax1.set_title(f"Signal Data vs Ground Truth (Section: {section_id}...)")
    ax1.set_ylabel("G-Force")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Ribbon Diagram
    draw_ribbons(ax2, 0.0, res_df['pred_smooth'], res_df['time'], "Smoothed AI", color_map)
    draw_ribbons(ax2, 1.0, res_df['pred_raw'], res_df['time'], "Raw AI Output", color_map)
    draw_ribbons(ax2, 2.0, res_df['truth'], res_df['time'], "Ground Truth", color_map)

    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Relative Time (Seconds)")
    ax2.set_title("Classification Pipeline Results")

    # Global Legend
    handles = [mpatches.Patch(color=color_map[c], label=c) for c in unique_classes]
    fig.legend(handles=handles, loc='lower center', ncol=len(unique_classes), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight")
    print(f"Saved plot to {save_name}")
    plt.show()
