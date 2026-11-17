"""
How the solid's pulse shape changes, section by section and across the
transitions between runs.

Reads what lifetime_vs_time.py saved (sums.pkl and lifetime_vs_time.json),
so run that first on the solid runs with small blocks, e.g. 5 files
(~3.5 minutes) per block. No data files are re-read here.

Outputs (in --output-dir):
  solid_shape_map.png    every block's average pulse divided by a reference
                         block (both normalised to the same total light),
                         as a colour map: time across, block down. White =
                         same shape as the reference; red = more light at
                         that time; blue = less. Run boundaries marked.
  solid_transitions.png  the last block of each run against the first block
                         of the next, as ratios to the reference
  solid_fit_params.png   tau_slow, slow fraction, intermediate tau and floor
                         share per block, with run boundaries

Usage (from the repo root):
    python analysis/solid_sections.py --lifetime-dir results/lifetime_solid5 \\
        --output-dir results/advisor/11_solid_sections
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SAMPLE_NS = 2.0
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9a5bd3", "#c0392b", "#555555"]


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load(lifetime_dir):
    """Blocks in time order: label, run, average pulse (unit area) and fit."""
    with open(Path(lifetime_dir) / "lifetime_vs_time.json") as f:
        summary = json.load(f)
    with open(Path(lifetime_dir) / "sums.pkl", "rb") as f:
        saved = pickle.load(f)
    max_jump = float(summary["settings"]["max_jump"])
    blocks = []
    for g, tot in zip(summary["groups"], saved["totals"]):
        n = tot["n"].get(max_jump, 0)
        if n < 2:
            continue
        M = tot["sum"][max_jump] / n
        blocks.append({"label": g["label"], "run": g["run"], "first": g["first"],
                       "M": M / M.sum(), "fit": g["fit"], "n": n})
    blocks.sort(key=lambda b: (b["run"], b["first"]))
    return blocks


def rebin(x, k):
    """Sum over blocks of k samples (a 1-D pulse, or one pulse per row)."""
    x = np.asarray(x)
    n = x.shape[-1] // k * k
    return x[..., :n].reshape(*x.shape[:-1], n // k, k).sum(axis=-1)


def ratio_to(blk, ref, k):
    """Block pulse / reference pulse in time bins of k samples; NaN where
    the reference holds almost no light (before the pulse starts)."""
    a, b = rebin(blk["M"], k), rebin(ref["M"], k)
    return np.where(b > 1e-4 * b.max(), a / np.where(b > 0, b, 1), np.nan)


def run_edges(blocks):
    """Index where each run starts, and its name."""
    edges = []
    for i, b in enumerate(blocks):
        if i == 0 or b["run"] != blocks[i - 1]["run"]:
            edges.append((i, b["run"]))
    return edges


def plot_map(blocks, ref, args, path):
    ratio = np.array([ratio_to(b, ref, args.bin) for b in blocks])
    t_us = np.arange(ratio.shape[1] + 1) * args.bin * SAMPLE_NS / 1000   # bin edges
    fig, ax = plt.subplots(figsize=(14, 0.12 * len(blocks) + 3))
    m = ax.pcolormesh(t_us, np.arange(len(blocks) + 1) - 0.5, ratio - 1, shading="flat",
                      cmap="RdBu_r", norm=TwoSlopeNorm(0, -args.range, args.range))
    for i, run in run_edges(blocks):
        ax.axhline(i - 0.5, color="#222222", linewidth=1)
        ax.text(t_us[-1] * 0.99, i, run, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
    ax.set_ylim(len(blocks) - 0.5, -0.5)
    ax.set_xlabel("time from start of charge window (μs)")
    ax.set_ylabel("block (time order, top = earliest)")
    ax.set_title(f"Pulse shape relative to {ref['label']} (both normalised to equal total light)")
    fig.colorbar(m, ax=ax, label="block ÷ reference − 1")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transitions(blocks, ref, args, path):
    edges = run_edges(blocks)
    pairs = [(blocks[i - 1], blocks[i]) for i, _ in edges[1:]]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4.5), squeeze=False, sharey=True)
    for ax, (a, b) in zip(axes[0], pairs):
        for blk, c, ls in ((a, "#2a78d6", "-"), (b, "#eb6834", "-")):
            r = ratio_to(blk, ref, args.bin)
            t = (np.arange(len(r)) + 0.5) * args.bin * SAMPLE_NS / 1000
            ax.plot(t, r, ls, color=c, linewidth=1.6, label=f"{blk['label']}  (τ {blk['fit']['tau_slow_ns']:.0f} ns)"
                    if blk["fit"] else blk["label"])
        ax.axhline(1, color="#222222", linewidth=0.8)
        ax.set_title(f"{a['run']} → {b['run']}")
        ax.set_xlabel("time (μs)")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0][0].set_ylabel(f"pulse ÷ {ref['label']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_params(blocks, path):
    keys = [("tau_slow_ns", "slow lifetime τ (ns)"), ("slow_frac", "slow fraction"),
            ("tau_inter_ns", "intermediate τ (ns)"), ("floor_share", "floor share of light")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 10), sharex=True)
    edges = run_edges(blocks)
    for (key, name), ax in zip(keys, axes):
        for j, (i0, run) in enumerate(edges):
            i1 = edges[j + 1][0] if j + 1 < len(edges) else len(blocks)
            xs = [i for i in range(i0, i1) if blocks[i]["fit"] and key in blocks[i]["fit"]]
            ax.plot(xs, [blocks[i]["fit"][key] for i in xs], "o", markersize=4,
                    color=RUN_COLORS[j % len(RUN_COLORS)], label=run)
            ax.axvline(i0 - 0.5, color="#999999", linewidth=0.8)
        ax.set_ylabel(name)
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, ncol=len(edges))
    axes[-1].set_xlabel("block (time order)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = load(args.lifetime_dir)
    ref = next((b for b in blocks if b["label"] == args.reference), None)
    if ref is None:
        ref = blocks[len(blocks) // 2]
        print(f"reference {args.reference} not found; using {ref['label']}")
    print(f"{len(blocks)} blocks; reference {ref['label']}")
    plot_map(blocks, ref, args, out / "solid_shape_map.png")
    plot_transitions(blocks, ref, args, out / "solid_transitions.png")
    plot_params(blocks, out / "solid_fit_params.png")
    print(f"plots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime_solid5")
    ap.add_argument("--output-dir", default="results/solid_sections")
    ap.add_argument("--reference", default="260520_1800:100-104", help="Block label used as the reference shape")
    ap.add_argument("--bin", type=int, default=25, help="Samples per time bin in the plots (25 = 50 ns)")
    ap.add_argument("--range", type=float, default=0.3, help="Colour scale: ± this fractional difference")
    main(ap.parse_args())
