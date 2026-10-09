"""
Track the argon's behaviour file by file across the whole May 20-21 dataset.

Each CoMPASS file covers a short stretch of time (roughly 40 s of a run), so
plotting a quantity per file, in file order, is a time series. Two
quantities are tracked, both from bright events (15-150 PE) after the
saturation and pileup cuts:

  - median psd_param: the prompt light fraction. It depends on the phase
    (gas / liquid / solid), so a phase change such as freezing shows up as
    a step or a drift.
  - median n_pe of the run's alpha band: how bright the Am-241 alphas look.
    With a stable gain (already checked), a change here means a change in
    how much light is produced or collected.

Useful for: checking whether a run is stable enough to treat as one
measurement, and seeing when a transition (e.g. freezing) happened.

Files are ordered by their sequence number from parse_run_info, not by
name (alphabetical order would put _10 before _2).

Usage (from the repo root; login node is fine):
    python analysis/stability_vs_time.py --output-dir results/stability
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}
NPE = np.arange(0, 151, 1.0)          # 1 PE bins
PSD = np.linspace(-0.1, 1.1, 121)     # 0.01 bins


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def hist_median(counts, edges):
    """Median from a histogram, interpolated within bins."""
    total = counts.sum()
    if total < 20:
        return np.nan
    cdf = np.concatenate([[0], np.cumsum(counts)]) / total
    return float(np.interp(0.5, cdf, edges))


def run_files(run_dir):
    files = list((Path(run_dir) / "hit").glob("*.parquet"))
    return sorted(files, key=lambda f: parse_run_info(f.stem)[1])


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}

    run_dirs = sorted(p for p in Path(args.data_root).glob("data_2605*") if (p / "hit").exists())
    cols = ["n_pe", "psd_param", "lowest_adc", "max_jump_pe"]
    lo_npe, hi_npe = args.bright

    runs = []
    for rd in run_dirs:
        files = run_files(rd)
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        phase = phase_db.get(run_base.replace("_", ""), "?")
        log.info(f"{rd.name} ({phase}): {len(files)} files")
        per_file = []          # (seq, 2D hist of bright events n_pe x psd)
        for f in files:
            t = pq.read_table(str(f), columns=cols)
            n_pe, psd, low, jump = (t.column(c).to_numpy() for c in cols)
            keep = (np.isfinite(psd) & (low >= args.sat_floor + args.sat_margin)
                    & (jump < args.max_jump) & (n_pe >= lo_npe) & (n_pe < hi_npe))
            h = np.histogram2d(n_pe[keep], psd[keep], bins=[NPE, PSD])[0]
            per_file.append((parse_run_info(f.stem)[1], h))
        total = sum(h for _, h in per_file)
        # the run's alpha band: peak of its bright-event PSD distribution
        prof = np.convolve(total.sum(axis=0), np.ones(3) / 3, mode="same")
        c = 0.5 * (PSD[:-1] + PSD[1:])[np.argmax(prof)]
        band = (c - args.band_half_width, c + args.band_half_width)
        j0, j1 = np.searchsorted(PSD, band)
        rows = []
        for seq, h in per_file:
            rows.append({
                "seq": int(seq),
                "n_bright": int(h.sum()),
                "median_psd": hist_median(h.sum(axis=0), PSD),
                "median_npe_band": hist_median(h[:, j0:j1].sum(axis=1), NPE),
            })
        runs.append({"run": run_base, "phase": phase, "band": [float(b) for b in band], "files": rows})

    # ---- plot: one shared x-axis, runs side by side in time order ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    x0 = 0
    for r in runs:
        x = x0 + np.arange(len(r["files"]))
        color = PHASE_COLOR.get(r["phase"], "#777777")
        for ax, key in ((ax1, "median_psd"), (ax2, "median_npe_band")):
            ax.plot(x, [row[key] for row in r["files"]], "o", markersize=3, color=color)
            ax.axvline(x0 - 0.5, color="#999999", linewidth=0.8)
        ax1.text(x0 + len(x) / 2, 1.02, f"{r['run']}\n{r['phase']}", transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=9)
        x0 += len(x)
    ax1.set_ylabel(f"median psd_param\n({lo_npe:g}-{hi_npe:g} PE, after cuts)")
    ax2.set_ylabel("median n_pe in the run's\nalpha band")
    ax2.set_xlabel("file number across the dataset (each file ≈ 40 s; runs in name order)")
    for ax in (ax1, ax2):
        style(ax)
    for phase, color in PHASE_COLOR.items():
        ax1.plot([], [], "o", color=color, label=phase)
    ax1.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "stability_vs_time.png", dpi=120)
    plt.close(fig)

    print(f"\n{'run':<13}{'phase':<8}{'band':>14}{'median PSD first/last':>24}{'alpha n_pe first/last':>24}")
    for r in runs:
        f = r["files"]
        k = max(1, min(5, len(f) // 4))   # average a few files at each end
        def avg(key, part):
            v = [row[key] for row in part if np.isfinite(row[key])]
            return np.mean(v) if v else np.nan
        print(f"{r['run']:<13}{r['phase']:<8}{r['band'][0]:>7.2f}-{r['band'][1]:<6.2f}"
              f"{avg('median_psd', f[:k]):>12.3f} / {avg('median_psd', f[-k:]):<9.3f}"
              f"{avg('median_npe_band', f[:k]):>12.1f} / {avg('median_npe_band', f[-k:]):<9.1f}")

    with open(out / "stability_vs_time.json", "w") as fh:
        json.dump(runs, fh, indent=1, default=float)
    log.info(f"done: {out / 'stability_vs_time.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/stability")
    ap.add_argument("--bright", type=float, nargs=2, default=[15, 150], help="n_pe range of 'bright' events")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=50)
    ap.add_argument("--max-jump", type=float, default=3.0)
    ap.add_argument("--band-half-width", type=float, default=0.08)
    main(ap.parse_args())
