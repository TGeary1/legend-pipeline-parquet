"""
Energy vs pulse shape, split by the pileup cut: shows what the cut removes.

A second pulse landing after the prompt window adds to the total charge but
not to the prompt charge, so pileup events sit at LOWER PSD and HIGHER
energy than the alphas they ride on. Plotting the removed events on their
own shows whether the cut is catching that population (and only that).

For each window (default: one stable stretch per phase) three panels of
PSD vs n_pe, after the saturation cut:
    all events | kept (max_jump_pe < cut) | removed (max_jump_pe >= cut)
plus one row of PSD distributions for alpha-energy events, kept vs removed.

Reads the hit tier only; fine on a login node.

Usage (from the repo root):
    python analysis/pileup_vs_psd.py --output-dir results/advisor/06_pileup
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

COLS = ["n_pe", "psd_param", "lowest_adc", "max_jump_pe"]
DEFAULT_WINDOWS = ["gas:260520_1239:0-42",
                   "liquid (early):260520_1340:0-8",
                   "solid (evening):260520_1800:70-172"]
LOG_NPE = np.logspace(np.log10(0.5), np.log10(500), 161)
PSD = np.linspace(-0.1, 1.1, 121)


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def window_files(data_root, run_glob, run_base, first, last):
    """Hit files of one run with sequence numbers first..last."""
    for rd in sorted(Path(data_root).glob(run_glob)):
        files = list((rd / "hit").glob("*.parquet"))
        if files and parse_run_info(files[0].stem)[0] == run_base:
            picked = [f for f in files if first <= parse_run_info(f.stem)[1] <= last]
            return sorted(picked, key=lambda f: parse_run_info(f.stem)[1])
    return []


def load(files, args):
    """n_pe, PSD and the pileup flag of all unsaturated events in the files."""
    parts = []
    for f in files:
        t = pq.read_table(str(f), columns=COLS)
        n, p, low, jump = (t.column(c).to_numpy() for c in COLS)
        ok = (low >= args.sat_floor + args.sat_margin) & np.isfinite(p)
        parts.append((n[ok], p[ok], jump[ok] >= args.max_jump))
    n, p, piled = (np.concatenate(x) for x in zip(*parts))
    return n, p, piled


def plot(windows, args, path):
    fig, axes = plt.subplots(len(windows), 4, figsize=(24, 5.2 * len(windows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for row, (label, n, p, piled) in zip(axes, windows):
        alpha = (n >= args.alpha_min) & (n <= args.alpha_max)
        for ax, sel, title in ((row[0], np.ones_like(piled), "all"),
                               (row[1], ~piled, f"kept (jump < {args.max_jump:g} PE)"),
                               (row[2], piled, f"removed (jump ≥ {args.max_jump:g} PE)")):
            h, _, _ = np.histogram2d(n[sel], p[sel], bins=(LOG_NPE, PSD))
            m = ax.pcolormesh(LOG_NPE, PSD, np.where(h.T > 0, h.T, np.nan), norm=LogNorm(), cmap="viridis")
            ax.set_xscale("log")
            ax.axvspan(args.alpha_min, args.alpha_max, color="#eb6834", alpha=0.08)
            ax.set_title(f"{label}: {title} ({sel.sum():,})")
            ax.set_xlabel("n_pe (photoelectrons)")
            ax.set_ylabel("PSD (prompt / total)")
            fig.colorbar(m, ax=ax, label="events")
        ax = row[3]
        for sel, color, name in ((alpha & ~piled, "#2a78d6", "kept"), (alpha & piled, "#eb6834", "removed")):
            ax.hist(p[sel], bins=PSD, density=True, histtype="step", color=color, linewidth=1.6,
                    label=f"{name} ({sel.sum():,})")
        ax.set_title(f"PSD, {args.alpha_min:g}–{args.alpha_max:g} PE")
        ax.set_xlabel("PSD (prompt / total)")
        ax.set_ylabel("normalised")
        ax.legend(frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    windows = []
    print(f"\n{'window':<22}{'events':>12}{'removed':>10}{'removed, alpha range':>22}"
          f"{'PSD kept':>10}{'PSD removed':>13}")
    for w in args.window or DEFAULT_WINDOWS:
        label, run_base, rng = w.split(":")
        first, last = (int(x) for x in rng.split("-"))
        files = window_files(args.data_root, args.run_glob, run_base, first, last)
        if not files:
            print(f"{label}: no files found for {run_base} {rng}")
            continue
        n, p, piled = load(files, args)
        windows.append((label, n, p, piled))
        alpha = (n >= args.alpha_min) & (n <= args.alpha_max)
        print(f"{label:<22}{len(n):>12,}{piled.mean():>10.1%}{piled[alpha].mean():>22.1%}"
              f"{np.median(p[alpha & ~piled]):>10.3f}{np.median(p[alpha & piled]):>13.3f}")
    plot(windows, args, out / "pileup_vs_psd.png")
    print(f"\nwrote {out / 'pileup_vs_psd.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--output-dir", default="results/pileup")
    ap.add_argument("--window", action="append", help="LABEL:RUN:FIRST-LAST; repeatable (default: one per phase)")
    ap.add_argument("--max-jump", type=float, default=8.0, help="Pileup cut, PE")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=50)
    ap.add_argument("--alpha-min", type=float, default=20, help="Alpha energy range for the PSD panel, PE")
    ap.add_argument("--alpha-max", type=float, default=150)
    main(ap.parse_args())
