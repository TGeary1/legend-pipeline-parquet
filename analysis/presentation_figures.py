"""
Two summary figures for presenting the lifetime results, made from the saved
outputs of lifetime_vs_time.py (sums.pkl and lifetime_vs_time.json). No data
files are re-read, so this runs in seconds on a login node.

 1. waveform_overlays.png - average alpha waveforms of chosen blocks,
    peak-normalised, log scale: the pulse-shape changes seen directly.
 2. quenching_test.png    - (slow/prompt) / tau against tau for every gas and
    solid block. Flat within a phase = the slow light is lost by quenching.

Usage (from the repo root):
    python analysis/presentation_figures.py --lifetime-dir results/lifetime
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_NS = 2.0
PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# (panel title, [(block label, legend text), ...])
PANELS = [
    ("Gas: purity swing", [("260520_1239:0-9", "start"),
                           ("260520_1239:20-29", "cleanest"),
                           ("260520_1239:40-42", "end")]),
    ("Liquid, then frozen", [("260520_1340:0-9", "liquid, early"),
                             ("260520_1340:40-49", "liquid, late"),
                             ("260520_1340:50-56", "frozen")]),
    ("Solid: evening vs morning", [("260520_1800:70-79", "evening"),
                                   ("260521_1023:0-9", "next morning"),
                                   ("260521_1023:90-99", "morning, ~1 h later")]),
]
EXCLUDE = {"260520_1447:50-59"}   # different fit solution in a collection dip
FROZEN = {"260520_1340:50-56"}    # liquid run, but solid-like pulse shape


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load(lifetime_dir, phases_path):
    """Block labels, phases, fits and average waveforms, in one dict per block."""
    with open(Path(lifetime_dir) / "lifetime_vs_time.json") as f:
        summary = json.load(f)
    with open(Path(lifetime_dir) / "sums.pkl", "rb") as f:
        saved = pickle.load(f)
    with open(phases_path) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    max_jump = float(summary["settings"]["max_jump"])

    blocks = {}
    for g, tot in zip(summary["groups"], saved["totals"]):
        n = tot["n"].get(max_jump, 0)
        blocks[g["label"]] = {
            "phase": phase_db.get(g["run"].replace("_", ""), "?"),
            "fit": g["fit"],
            "mean_wf": tot["sum"][max_jump] / n if n else None,
        }
    return blocks


def plot_overlays(blocks, path):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.8), sharey=True)
    for ax, (title, members) in zip(axes, PANELS):
        for (label, text), color in zip(members, LINE_COLORS):
            b = blocks.get(label)
            if b is None or b["mean_wf"] is None:
                continue
            wf = b["mean_wf"]
            t_us = np.arange(len(wf)) * SAMPLE_NS / 1000
            extra = f", τ = {b['fit']['tau_slow_ns'] / 1000:.2f} μs" if b["fit"] else ""
            ax.semilogy(t_us, np.clip(wf / wf.max(), 1e-4, None), color=color, linewidth=1.3,
                        label=f"{text}{extra}")
        ax.set_title(title)
        ax.set_xlabel("time from start of charge window (μs)")
        ax.set_ylim(1e-3, 1.5)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0].set_ylabel("average alpha waveform (peak = 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_quenching_test(blocks, path):
    """(slow/prompt)/tau vs tau. Quenching predicts a flat line within a phase.
    The dashed curves show the alternative: slow/prompt fixed while tau
    changes, i.e. ratio proportional to 1/tau, drawn through each phase's
    shortest-lifetime block."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    points = {"gas": [], "solid": []}
    for label, b in blocks.items():
        f = b["fit"]
        if f is None or label in EXCLUDE or (b["phase"] == "liquid" and label not in FROZEN):
            continue
        tau_us = f["tau_slow_ns"] / 1000
        s = f["slow_frac"]
        y = s / (1 - s) / tau_us
        if label in FROZEN:
            ax.plot(tau_us, y, "D", color=PHASE_COLOR["liquid"], markersize=7,
                    label="end of liquid run (frozen)")
        else:
            points.setdefault(b["phase"], []).append((tau_us, y))
            ax.plot(tau_us, y, "o", color=PHASE_COLOR.get(b["phase"], "#777777"), markersize=5)

    for phase, pts in points.items():
        if len(pts) < 2:
            continue
        tau0, y0 = min(pts)                       # shortest-lifetime block
        tau_max = max(t for t, _ in pts)
        t = np.linspace(tau0, tau_max, 100)
        ax.plot(t, y0 * tau0 / t, "--", color=PHASE_COLOR[phase], linewidth=1, alpha=0.7)
        ax.plot([], [], "o", color=PHASE_COLOR[phase], label=phase)
    ax.plot([], [], "--", color="#777777", label="expected if slow light did not follow τ")

    ax.set_xlabel("triplet lifetime τ (μs)")
    ax.set_ylabel("(slow / prompt) ÷ τ   (per μs)")
    ax.set_title("Quenching test: flat within a phase as τ changes")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    blocks = load(args.lifetime_dir, args.phases)
    out = Path(args.output_dir or args.lifetime_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_overlays(blocks, out / "waveform_overlays.png")
    plot_quenching_test(blocks, out / "quenching_test.png")
    print(f"wrote {out / 'waveform_overlays.png'} and {out / 'quenching_test.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime",
                    help="Output folder of lifetime_vs_time.py (holds sums.pkl and the JSON)")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default=None, help="Default: same as --lifetime-dir")
    main(ap.parse_args())
