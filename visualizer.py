"""
Attention Drop Detector — Visualizer
=====================================
Generates a three-panel chart from an analysis result:
  1. Attention score over time (color-coded by label)
  2. Per-window feature signals (motion, audio, cuts)
  3. Color-coded engagement strip

Usage (standalone):
  python visualizer.py result.json
  python visualizer.py result.json --out chart.png --show
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

from analyzer import Drop, Result, Window


# ── Style ─────────────────────────────────────────────────────────────────────
BG      = "#0d0d0f"
SURFACE = "#141417"
BORDER  = "#232328"
TEXT    = "#e5e5e5"
MUTED   = "#6b7280"

C_ENGAGING = "#4ade80"
C_NEUTRAL  = "#fbbf24"
C_DROP     = "#f87171"
C_MOTION   = "#818cf8"
C_AUDIO    = "#22d3ee"
C_CUTS     = "#fb923c"


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   MUTED,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "font.size":         9,
    })


def _fmt_time(x, _=None) -> str:
    m, s = divmod(int(max(x, 0)), 60)
    return f"{m}:{s:02d}"


def _colored_line(ax, x, y, labels) -> None:
    """Draws the score line with each segment colored by its window label."""
    color_map = {
        "engaging": C_ENGAGING,
        "neutral":  C_NEUTRAL,
        "drop":     C_DROP,
    }
    for i in range(len(x) - 1):
        c = color_map.get(labels[i], C_NEUTRAL)
        ax.plot(x[i:i+2], y[i:i+2], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter(x[i], y[i], color=c, s=22, zorder=5)
    if len(x):
        c = color_map.get(labels[-1], C_NEUTRAL)
        ax.scatter(x[-1], y[-1], color=c, s=22, zorder=5)


def plot(
    result: Result,
    save_path: str = "attention_chart.png",
    show: bool = False,
) -> None:
    """Generate and save the full attention visualization."""
    windows = result.windows
    if not windows:
        print("No windows to plot.")
        return

    _apply_style()

    times  = np.array([(w.start + w.end) / 2 for w in windows])
    scores = np.array([w.score               for w in windows])
    labels = [w.label                        for w in windows]

    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Attention Drop Detector",
        fontsize=15, fontweight="bold", color=TEXT, y=0.975
    )

    gs     = fig.add_gridspec(3, 1, hspace=0.52, height_ratios=[4, 2.2, 0.7])
    ax_s   = fig.add_subplot(gs[0])          # score line
    ax_f   = fig.add_subplot(gs[1], sharex=ax_s)   # feature bars
    ax_bar = fig.add_subplot(gs[2], sharex=ax_s)   # color strip

    # ── Panel 1: score timeline ───────────────────────────────────────────────
    _colored_line(ax_s, times, scores, labels)

    # Shade drop windows
    for w in windows:
        if w.label == "drop":
            ax_s.axvspan(w.start, w.end, color=C_DROP, alpha=0.07)

    # Threshold lines
    ax_s.axhline(0.35, color=C_DROP,     lw=0.8, ls="--", alpha=0.55)
    ax_s.axhline(0.65, color=C_ENGAGING, lw=0.8, ls="--", alpha=0.55)
    ax_s.text(times[-1] + 1, 0.36, "drop zone",     color=C_DROP,     fontsize=7.5, va="bottom")
    ax_s.text(times[-1] + 1, 0.66, "engaging zone", color=C_ENGAGING, fontsize=7.5, va="bottom")

    ax_s.set_ylim(-0.05, 1.12)
    ax_s.set_ylabel("Attention Score", fontsize=9, color=MUTED)
    ax_s.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_s.grid(True, axis="y")
    ax_s.set_title("Score Timeline", fontsize=10, color=MUTED, pad=7)

    legend = [
        mpatches.Patch(color=C_ENGAGING, label="Engaging"),
        mpatches.Patch(color=C_NEUTRAL,  label="Neutral"),
        mpatches.Patch(color=C_DROP,     label="Drop"),
    ]
    ax_s.legend(
        handles=legend, loc="upper right", fontsize=8,
        framealpha=0.25, facecolor=SURFACE, labelcolor=TEXT
    )

    # Annotate the worst drop
    if result.drops:
        worst_win  = min(windows, key=lambda w: w.score)
        worst_t    = (worst_win.start + worst_win.end) / 2
        ax_s.annotate(
            f"worst drop\n{worst_win.score:.0%}",
            xy=(worst_t, worst_win.score),
            xytext=(worst_t, worst_win.score + 0.18),
            fontsize=7.5, color=C_DROP, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_DROP, lw=0.8),
        )

    # ── Panel 2: feature signals ──────────────────────────────────────────────
    max_cuts = max((w.cuts for w in windows), default=1) or 1
    motion   = [w.motion          for w in windows]
    audio    = [w.audio           for w in windows]
    cuts_n   = [w.cuts / max_cuts for w in windows]

    bw = result.window_sec * 0.27
    ax_f.bar(times - bw, motion, width=bw, color=C_MOTION, alpha=0.85, label="Motion")
    ax_f.bar(times,      audio,  width=bw, color=C_AUDIO,  alpha=0.85, label="Audio energy")
    ax_f.bar(times + bw, cuts_n, width=bw, color=C_CUTS,   alpha=0.85, label="Cut frequency")

    ax_f.set_ylim(0, 1.18)
    ax_f.set_ylabel("Signal (norm.)", fontsize=9, color=MUTED)
    ax_f.set_title("Feature Signals per Window", fontsize=10, color=MUTED, pad=7)
    ax_f.grid(True, axis="y")
    ax_f.legend(
        loc="upper right", fontsize=8,
        framealpha=0.25, facecolor=SURFACE, labelcolor=TEXT
    )

    # ── Panel 3: color strip ──────────────────────────────────────────────────
    color_map = {"engaging": C_ENGAGING, "neutral": C_NEUTRAL, "drop": C_DROP}
    for w in windows:
        ax_bar.barh(
            0, w.end - w.start, left=w.start,
            height=1, color=color_map[w.label], align="center"
        )
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Time", fontsize=9, color=MUTED)
    ax_bar.set_title("Engagement Map", fontsize=9, color=MUTED, pad=5)

    # Shared x-axis formatting
    for ax in [ax_s, ax_f, ax_bar]:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))

    plt.savefig(save_path, bbox_inches="tight", facecolor=BG, dpi=140)
    print(f"  Chart saved: {save_path}")

    if show:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention analysis results")
    parser.add_argument("result_json", help="Path to result.json from analyzer.py")
    parser.add_argument("--out",  "-o", default="attention_chart.png")
    parser.add_argument("--show", "-s", action="store_true")
    args = parser.parse_args()

    with open(args.result_json) as f:
        data = json.load(f)

    windows = [Window(**w) for w in data["windows"]]
    drops   = [Drop(**d)   for d in data["drops"]]
    result  = Result(
        video      = data["video"],
        duration   = data["duration"],
        window_sec = data["window_sec"],
        windows    = windows,
        drops      = drops,
        summary    = data["summary"],
    )

    plot(result, save_path=args.out, show=args.show)
