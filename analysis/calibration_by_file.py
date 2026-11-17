"""
Check the single-photoelectron calibration file by file.

Each run was calibrated once, from all its files together. If the PMT gain
drifted within a run, early and late files would be converted to PE with
the wrong scale. This fits the 1 PE and 2 PE peaks of every file's n_pe
spectrum (n_pe = charge / that run's gain):

    peak 1 position     should be 1.00 if the run's gain fits this file
    peak spacing        the file's own gain relative to the run's (1.00 = same);
                        a spacing of peaks, so a constant offset in the
                        charge cancels
    peak 1 width        PMT resolution; a change hints at noise or threshold changes

Reads only the n_pe column of the hit tier; fine on a login node.

Usage (from the repo root):
    python analysis/calibration_by_file.py --output-dir results/advisor/10_calibration
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}
BINS = np.linspace(0.4, 3.6, 321)


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def gauss(x, a, mu, sig):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def peak_position(x, counts, guess, half_width):
    """Gaussian fit over a narrow window around the local maximum nearest
    `guess`. Fitting only the top of each peak keeps the fit away from the
    overlapping neighbours and the background under the valleys."""
    near = (x > guess - 0.35) & (x < guess + 0.35)
    if counts[near].sum() < 300:
        return None, None
    top = x[near][np.argmax(counts[near])]
    sel = (x > top - half_width) & (x < top + half_width)
    try:
        p, _ = curve_fit(gauss, x[sel], counts[sel], p0=[counts[sel].max(), top, half_width],
                         sigma=np.sqrt(np.maximum(counts[sel], 1)), maxfev=5000)
    except (RuntimeError, ValueError):
        return None, None
    return float(p[1]), abs(float(p[2]))


def file_peaks(path):
    """1 and 2 PE peak positions of one file's n_pe spectrum. Their spacing
    is the file's gain relative to the run's (the 3 PE peak is too small
    per file to add precision). Returns (1 PE position, 1 PE width, spacing)."""
    n = pq.read_table(str(path), columns=["n_pe"]).column("n_pe").to_numpy()
    counts, edges = np.histogram(n, bins=BINS)
    x = 0.5 * (edges[:-1] + edges[1:])
    ks, mus, sig1 = [], [], np.nan
    for k, hw in ((1, 0.12), (2, 0.15)):
        mu, sig = peak_position(x, counts, guess=k * (mus[0] if mus else 1.0), half_width=hw)
        if mu is None:
            break
        ks.append(k)
        mus.append(mu)
        if k == 1:
            sig1 = sig
    if len(mus) < 2:
        return np.nan, np.nan, np.nan
    slope, _ = np.polyfit(ks, mus, 1)
    return mus[0], sig1, slope


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}

    runs = []
    for rd in sorted(Path(args.data_root).glob(args.run_glob)):
        files = sorted((rd / "hit").glob("*.parquet"), key=lambda f: parse_run_info(f.stem)[1])
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        phase = phase_db.get(run_base.replace("_", ""))
        if phase is None or any(r[0] == run_base for r in runs):
            continue
        res = np.array([file_peaks(f) for f in files])
        runs.append((run_base, phase, res))
        print(f"{run_base}: {len(files)} files fitted")

    print(f"\n{'run':<14}{'phase':<8}{'peak 1: mean':>13}{'spread':>8}{'first 10%':>11}{'last 10%':>10}"
          f"{'gain ratio: mean':>18}{'spread':>8}{'first 10%':>11}{'last 10%':>10}")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    x0 = 0
    for run_base, phase, res in runs:
        mu1, s1, ratio = res.T
        x = x0 + np.arange(len(res))
        k = max(1, len(res) // 10)
        print(f"{run_base:<14}{phase:<8}{np.nanmean(mu1):>13.4f}{np.nanstd(mu1):>8.4f}"
              f"{np.nanmean(mu1[:k]):>11.4f}{np.nanmean(mu1[-k:]):>10.4f}"
              f"{np.nanmean(ratio):>18.4f}{np.nanstd(ratio):>8.4f}"
              f"{np.nanmean(ratio[:k]):>11.4f}{np.nanmean(ratio[-k:]):>10.4f}")
        c = PHASE_COLOR.get(phase, "#777777")
        for ax, y in zip(axes, (mu1, ratio, s1)):
            ax.plot(x, y, "o", color=c, markersize=3)
            ax.axvline(x0 - 0.5, color="#999999", linewidth=0.8)
        axes[0].text(x0 + len(res) / 2, 1.02, f"{run_base}\n{phase}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=9)
        x0 += len(res)
    for ax, ref in zip(axes[:2], (1.0, 1.0)):
        ax.axhline(ref, color="#222222", linestyle="--", linewidth=1)
        for d in (-0.01, 0.01):
            ax.axhline(ref + d, color="#999999", linestyle=":", linewidth=1)
    axes[0].set_ylabel("1 PE peak position\n(n_pe; 1.00 = run calibration fits)")
    axes[1].set_ylabel("peak spacing\n(file gain ÷ run gain)")
    axes[2].set_ylabel("1 PE peak width σ\n(PE)")
    axes[2].set_xlabel("file number across the dataset (each file ≈ 40 s; runs in name order); dotted: ±1%")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "calibration_by_file.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out / 'calibration_by_file.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/calibration_by_file")
    main(ap.parse_args())
