"""
Pileup cut as a fraction of the event's own light, compared with the
current fixed 8 PE cut.

The current tag, max_jump_pe, is the largest rise of the waveform within
20 ns anywhere after 0.6 us into the charge window, in units of the
single-photoelectron pulse height: roughly how many photons arrived
together. The fixed cut keeps events with max_jump_pe < 8.

A relative cut asks instead: did any single 20 ns step bring more than X%
of the event's total light?   max_jump_pe / n_pe < X
It scales with energy, so bright events are not cut more often just
because their slow light is denser. Its weak point is dim events: 10% of
a 20 PE alpha is 2 PE, which a pair of slow-light photons can reach.
Hence --floor-pe: a step must also exceed this many PE to count as pileup.

Reads the hit tier only; fine on a login node.

Usage (from the repo root):
    python analysis/pileup_relative.py --output-dir results/advisor/08_pileup_relative
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
LOG_NPE = np.logspace(np.log10(5), np.log10(500), 81)
FRAC = np.linspace(0, 0.6, 121)


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
    """n_pe, PSD and max_jump_pe of unsaturated events above --min-pe."""
    parts = []
    for f in files:
        t = pq.read_table(str(f), columns=COLS)
        n, p, low, jump = (t.column(c).to_numpy() for c in COLS)
        ok = (low >= args.sat_floor + args.sat_margin) & np.isfinite(p) & (n >= args.min_pe)
        parts.append((n[ok], p[ok], jump[ok]))
    return (np.concatenate(x) for x in zip(*parts))


def cuts(n, jump, args):
    """Boolean 'removed' mask for each cut being compared."""
    rel = jump / n
    out = {f"fixed {args.max_jump:g} PE": jump >= args.max_jump}
    for f in args.fractions:
        out[f"relative {f:.0%}"] = (rel >= f) & (jump >= args.floor_pe)
    return out


def plot(windows, args, path):
    fig, axes = plt.subplots(len(windows), 3, figsize=(21, 5.2 * len(windows)), squeeze=False)
    colors = ["#555555", "#2a78d6", "#eb6834", "#1baf7a", "#9a5bd3"]
    for row, (label, n, p, jump) in zip(axes, windows):
        # 1: relative jump vs energy, with the cut lines
        ax = row[0]
        h, _, _ = np.histogram2d(n, jump / n, bins=(LOG_NPE, FRAC))
        m = ax.pcolormesh(LOG_NPE, FRAC, np.where(h.T > 0, h.T, np.nan), norm=LogNorm(), cmap="viridis")
        ax.set_xscale("log")
        xs = np.logspace(np.log10(5), np.log10(500), 200)
        ax.plot(xs, args.max_jump / xs, "--", color="#222222", linewidth=1.5, label=f"fixed {args.max_jump:g} PE")
        for f, c in zip(args.fractions, colors[1:]):
            ax.plot(xs, np.where(xs * f >= args.floor_pe, f, args.floor_pe / xs), ":", color=c,
                    linewidth=1.8, label=f"relative {f:.0%}")
        ax.set_ylim(0, 0.6)
        ax.set_title(f"{label}: largest 20 ns step ÷ total light")
        ax.set_xlabel("n_pe (photoelectrons)")
        ax.set_ylabel("max_jump_pe / n_pe")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.colorbar(m, ax=ax, label="events")

        # 2: fraction removed vs energy, per cut
        ax = row[1]
        idx = np.digitize(n, LOG_NPE) - 1
        centres = np.sqrt(LOG_NPE[:-1] * LOG_NPE[1:])
        for (name, rem), c in zip(cuts(n, jump, args).items(), colors):
            num = np.bincount(idx, weights=rem, minlength=len(centres))[:len(centres)]
            den = np.bincount(idx, minlength=len(centres))[:len(centres)]
            frac = np.where(den > 50, num / np.maximum(den, 1), np.nan)
            ax.semilogx(centres, frac, color=c, linewidth=1.8, label=name)
        ax.set_ylim(0, None)
        ax.set_title("fraction removed vs energy")
        ax.set_xlabel("n_pe (photoelectrons)")
        ax.set_ylabel("removed fraction")
        ax.legend(frameon=False, fontsize=9)
        style(ax)

        # 3: PSD of what each cut removes, alpha energies only
        ax = row[2]
        alpha = (n >= args.alpha_min) & (n <= args.alpha_max)
        bins = np.linspace(-0.1, 1.1, 121)
        ax.hist(p[alpha], bins=bins, density=True, histtype="stepfilled", color="#dddddd", label="all alphas")
        for (name, rem), c in zip(cuts(n, jump, args).items(), colors):
            sel = alpha & rem
            if sel.sum() > 50:
                ax.hist(p[sel], bins=bins, density=True, histtype="step", color=c, linewidth=1.5,
                        label=f"removed, {name}")
        ax.set_title(f"PSD of removed events, {args.alpha_min:g}–{args.alpha_max:g} PE")
        ax.set_xlabel("PSD (prompt / total)")
        ax.set_ylabel("normalised")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    windows = []
    print(f"\n{'window':<20}{'cut':<16}{'removed':>9}{'removed, alpha':>16}"
          f"{'PSD kept':>10}{'PSD removed':>13}{'removed in band':>17}")
    for w in args.window or DEFAULT_WINDOWS:
        label, run_base, rng = w.split(":")
        first, last = (int(x) for x in rng.split("-"))
        files = window_files(args.data_root, args.run_glob, run_base, first, last)
        if not files:
            print(f"{label}: no files found for {run_base} {rng}")
            continue
        n, p, jump = load(files, args)
        windows.append((label, n, p, jump))
        alpha = (n >= args.alpha_min) & (n <= args.alpha_max)
        # alpha band: +-0.08 around the PSD peak of alpha-energy events
        counts, edges = np.histogram(p[alpha], bins=np.linspace(-0.1, 1.1, 241))
        peak = 0.5 * (edges[:-1] + edges[1:])[np.argmax(counts)]
        band = alpha & (np.abs(p - peak) < 0.08)
        for name, rem in cuts(n, jump, args).items():
            pk = np.median(p[alpha & ~rem]) if (alpha & ~rem).any() else np.nan
            pr = np.median(p[alpha & rem]) if (alpha & rem).any() else np.nan
            print(f"{label:<20}{name:<16}{rem.mean():>9.1%}{rem[alpha].mean():>16.1%}"
                  f"{pk:>10.3f}{pr:>13.3f}{rem[band].mean():>17.1%}")
        print()
    plot(windows, args, out / "pileup_relative.png")
    print(f"wrote {out / 'pileup_relative.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--output-dir", default="results/pileup_relative")
    ap.add_argument("--window", action="append", help="LABEL:RUN:FIRST-LAST; repeatable (default: one per phase)")
    ap.add_argument("--max-jump", type=float, default=8.0, help="The current fixed cut, PE")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="Relative cuts to compare: largest 20 ns step as a fraction of total light")
    ap.add_argument("--floor-pe", type=float, default=3.0,
                    help="A step must also be at least this many PE to count as pileup")
    ap.add_argument("--min-pe", type=float, default=5.0, help="Ignore events dimmer than this")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    ap.add_argument("--alpha-min", type=float, default=20)
    ap.add_argument("--alpha-max", type=float, default=150)
    main(ap.parse_args())
