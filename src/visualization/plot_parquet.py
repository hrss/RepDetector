#!/usr/bin/env python3
"""
Plot a unified-format canonical session (.parquet + .meta.json).

Usage:
    python plot_session.py path/to/session.parquet
    python plot_session.py path/to/session.parquet -o out.png
    python plot_session.py path/to/session.parquet --lowpass 5
    python plot_session.py path/to/session.parquet --time-range 10 30
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.data_ingestion.session_io import SessionMeta, read_session


# ---------------------------------------------------------------------------
# Optional low-pass filter
# ---------------------------------------------------------------------------

def lowpass_filter(df, cutoff_hz: float, sample_rate_hz: int, order: int = 4):
    """Zero-phase Butterworth low-pass on all 6 IMU channels."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        raise SystemExit("scipy required for --lowpass. Install with: pip install scipy")

    nyq = sample_rate_hz / 2.0
    if cutoff_hz >= nyq:
        print(f"Warning: cutoff {cutoff_hz} Hz >= Nyquist {nyq:.1f} Hz; skipping filter.",
              file=sys.stderr)
        return df

    b, a = butter(order, cutoff_hz / nyq, btype="low")
    out = df.copy()
    for ch in ("ax", "ay", "az", "gx", "gy", "gz"):
        out[ch] = filtfilt(b, a, df[ch].to_numpy()).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Segment palette
# ---------------------------------------------------------------------------

def _make_palette(segments: list) -> dict[str, str]:
    """One distinct color per unique non-Rest segment name."""
    unique_names = []
    for s in segments:
        if s.name.upper() != "REST" and s.name not in unique_names:
            unique_names.append(s.name)
    palette: dict[str, str] = {"Rest": "#cccccc", "REST": "#cccccc"}
    for i, name in enumerate(unique_names):
        hue = i / max(len(unique_names), 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.65)
        palette[name] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return palette


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_session(
    df,
    meta: SessionMeta,
    out_path: Path,
    *,
    lowpass_hz: float | None = None,
    filter_order: int = 4,
    time_range: tuple[float, float] | None = None,
) -> None:
    # Apply optional filtering BEFORE slicing the time range, so the filter
    # has the full signal context (filtfilt edge effects shrink to the
    # session boundaries, not the view boundaries).
    if lowpass_hz is not None:
        df = lowpass_filter(df, lowpass_hz, meta.native_sample_rate_hz, filter_order)
        print(f"Applied Butterworth low-pass: order={filter_order}, cutoff={lowpass_hz} Hz")

    if time_range is not None:
        t0, t1 = time_range
        mask = (df["t_sec"] >= t0) & (df["t_sec"] <= t1)
        df = df.loc[mask].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples in time range [{t0}, {t1}]s")

    t = df["t_sec"].to_numpy()
    palette = _make_palette(meta.segments)

    channels = [
        ("ax", "Accel X", "g", "#1f77b4"),
        ("ay", "Accel Y", "g", "#1f77b4"),
        ("az", "Accel Z", "g", "#1f77b4"),
        ("gx", "Gyro X",  "rad/s", "#d62728"),
        ("gy", "Gyro Y",  "rad/s", "#d62728"),
        ("gz", "Gyro Z",  "rad/s", "#d62728"),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)

    for ax, (col, label, unit, color) in zip(axes, channels):
        # Segment overlays (drawn behind the trace).
        for seg in meta.segments:
            seg_color = palette.get(seg.name, "#eeeeee")
            is_rest = seg.name.upper() == "REST"
            alpha = 0.15 if is_rest else 0.25
            ax.axvspan(seg.start, seg.end, color=seg_color, alpha=alpha, linewidth=0)

            # Label segment names on the top subplot only, above the axes.
            if ax is axes[0] and not is_rest:
                mid = (seg.start + seg.end) / 2
                ax.text(
                    mid, 1.02, seg.name,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=7, rotation=30, clip_on=False,
                    color="#333333",
                )

        # The signal itself.
        ax.plot(t, df[col].to_numpy(), linewidth=0.7, color=color)

        # Rep tick marks per subplot.
        for seg in meta.segments:
            rep_color = palette.get(seg.name, "#999999")
            for rep in seg.reps:
                ax.axvline(rep.t, color=rep_color, linewidth=0.6, alpha=0.6, zorder=1.5)

        ax.set_ylabel(f"{label}\n({unit})", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)

    # Rep numbers on the top subplot, restarted per segment.
    for seg in meta.segments:
        if not seg.reps:
            continue
        sorted_reps = sorted(seg.reps, key=lambda r: r.t)
        seg_color = palette.get(seg.name, "#666")
        for j, rep in enumerate(sorted_reps):
            axes[0].text(
                rep.t, 1.13, str(j + 1),
                transform=axes[0].get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=6, color=seg_color, clip_on=False,
            )

    axes[-1].set_xlabel("Session time (s)")

    # Legend for segments.
    non_rest_palette = {k: v for k, v in palette.items() if k.upper() != "REST"}
    if non_rest_palette:
        handles = [
            mpatches.Patch(color=c, alpha=0.6, label=n)
            for n, c in non_rest_palette.items()
        ]
        total_reps = sum(len(s.reps) for s in meta.segments)
        if total_reps > 0:
            handles.append(mpatches.Patch(color="white", label=f"reps (n={total_reps})"))
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 7),
            fontsize=8,
            framealpha=0.8,
            bbox_to_anchor=(0.5, 0.0),
        )
        bottom_margin = 0.07 + 0.02 * ((len(handles) - 1) // 7)
    else:
        bottom_margin = 0.04

    # Title with the most diagnostic info up front.
    crown_str = f", crown={meta.crown}" if meta.crown else ""
    filter_note = f"  •  LP {lowpass_hz} Hz" if lowpass_hz else ""
    range_note = (
        f"  •  view {time_range[0]:.1f}-{time_range[1]:.1f}s"
        if time_range else ""
    )
    title_line1 = (
        f"{meta.session_id}  •  {meta.device}"
        f"{f' ({meta.device_model})' if meta.device_model else ''}"
        f"  •  wrist={meta.wrist}{crown_str}"
    )
    title_line2 = (
        f"{meta.n_samples:,} samples  •  {meta.duration_sec:.2f}s  •  "
        f"{meta.native_sample_rate_hz} Hz  •  {len(meta.segments)} segments"
        f"{filter_note}{range_note}"
    )
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=10)

    fig.tight_layout(rect=[0, bottom_margin, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Plot a unified canonical session.")
    p.add_argument("parquet", type=Path, help="Path to session .parquet")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output PNG path (default: alongside the parquet)")
    p.add_argument("--lowpass", type=float, default=None, metavar="HZ",
                   help="Apply Butterworth low-pass at this cutoff (e.g. 5)")
    p.add_argument("--filter-order", type=int, default=4)
    p.add_argument("--time-range", nargs=2, type=float, default=None,
                   metavar=("START", "END"),
                   help="Restrict view to this time window in seconds")
    args = p.parse_args()

    df, meta = read_session(args.parquet)
    out_path = args.out or args.parquet.with_suffix(".png")

    plot_session(
        df, meta, out_path,
        lowpass_hz=args.lowpass,
        filter_order=args.filter_order,
        time_range=tuple(args.time_range) if args.time_range else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())