"""
Track the argon's behaviour file by file across the whole May 20-21 dataset.

Each CoMPASS file covers a short stretch of time (roughly 40 s of a run), so
plotting a quantity per file, in file order, is a time series. Tracked,
from bright events (15-150 PE) after the saturation and pileup cuts:

  - median psd_param: the prompt light fraction. It depends on the phase
    (gas / liquid / solid), so a phase change such as freezing shows up as
    a step or a drift.
  - median n_pe of the run's alpha band: how bright the Am-241 alphas look.
    With a stable gain (already checked), a change here means a change in
    how much light is produced or collected.
  - the same split into prompt light (psd_param x n_pe) and slow light
    ((1 - psd_param) x n_pe). If both change by the same factor, the light
    reaching the PMT changed (optics, collection). If only the slow light
    changes, something acted on the triplet state (purity, temperature,
    phase).
  - alpha fraction: alpha-band events per recorded event. The recorded
    rate is fixed by the DAQ (it keeps ~10% of triggers), so the absolute
    rate says nothing about the argon; the fraction does. Read it next to
    the brightness: only alphas above the --bright threshold (15 PE) are
    counted, so plain dimming lowers the fraction too. A drop with
    unchanged brightness means alphas are being lost (blocked, or moved
    out of the PSD band).
  - pileup fraction: share of events failing the pileup cut. It rises with
    the true trigger rate, which the recorded rate hides.

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
PSD = np.linspace(-0.1, 1.1, 121)     # 0.01 bins


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def run_files(run_dir):
    files = list((Path(run_dir) / "hit").glob("*.parquet"))
    return sorted(files, key=lambda f: parse_run_info(f.stem)[1])


def file_stats(f, args):
    """One file: the bright events that pass the cuts, plus event counts."""
    cols = ["n_pe", "psd_param", "lowest_adc", "max_jump_pe"]
    t = pq.read_table(str(f), columns=cols)
    n_pe, psd, low, jump = (t.column(c).to_numpy() for c in cols)
    lo_npe, hi_npe = args.bright
    valid = np.isfinite(psd) & np.isfinite(n_pe)
    unsaturated = low >= args.sat_floor + args.sat_margin
    clean = jump < args.max_jump
    bright = (n_pe >= lo_npe) & (n_pe < hi_npe)
    keep = valid & unsaturated & clean & bright
    return {
        "seq": int(parse_run_info(f.stem)[1]),
        "n_events": int(len(n_pe)),
        "pileup_fraction": float(np.mean(~clean[valid])) if valid.any() else np.nan,
        "n_pe": n_pe[keep].astype(np.float32),
        "psd": psd[keep].astype(np.float32),
    }


def median_or_nan(v, min_n=20):
    return float(np.median(v)) if len(v) >= min_n else np.nan


def find_band(per_file, half_width):
    """The run's alpha band: the peak of its bright-event PSD distribution."""
    psd_all = np.concatenate([d["psd"] for d in per_file])
    counts, _ = np.histogram(psd_all, bins=PSD)
    prof = np.convolve(counts, np.ones(3) / 3, mode="same")
    c = 0.5 * (PSD[:-1] + PSD[1:])[np.argmax(prof)]
    return float(c - half_width), float(c + half_width)


def summarize(d, band):
    in_band = (d["psd"] >= band[0]) & (d["psd"] < band[1])
    n_pe, psd = d["n_pe"][in_band], d["psd"][in_band]
    return {
        "seq": d["seq"],
        "n_events": d["n_events"],
        "n_bright": int(len(d["psd"])),
        "median_psd": median_or_nan(d["psd"]),
        "median_npe_band": median_or_nan(n_pe),
        "median_prompt_pe": median_or_nan(psd * n_pe),
        "median_slow_pe": median_or_nan((1 - psd) * n_pe),
        "alpha_fraction": float(in_band.sum() / d["n_events"]) if d["n_events"] else np.nan,
        "pileup_fraction": d["pileup_fraction"],
    }


PANELS = [
    ("median_psd", "median psd_param\n(bright, after cuts)"),
    ("median_npe_band", "alpha brightness\n(median n_pe in band)"),
    (("median_prompt_pe", "median_slow_pe"), "alpha prompt (filled)\nand slow (open) PE"),
    ("alpha_fraction", "alpha-band events\nper recorded event"),
    ("pileup_fraction", "pileup fraction\n(max_jump_pe >= cut)"),
]


def plot(runs, path):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(15, 3 * len(PANELS)), sharex=True)
    x0 = 0
    for r in runs:
        x = x0 + np.arange(len(r["files"]))
        color = PHASE_COLOR.get(r["phase"], "#777777")
        for ax, (key, _) in zip(axes, PANELS):
            if isinstance(key, tuple):
                ax.plot(x, [row[key[0]] for row in r["files"]], "o", markersize=3, color=color)
                ax.plot(x, [row[key[1]] for row in r["files"]], "o", markersize=3, color=color,
                        markerfacecolor="none", markeredgewidth=0.8)
            else:
                ax.plot(x, [row[key] for row in r["files"]], "o", markersize=3, color=color)
            ax.axvline(x0 - 0.5, color="#999999", linewidth=0.8)
        axes[0].text(x0 + len(x) / 2, 1.02, f"{r['run']}\n{r['phase']}",
                     transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
        x0 += len(x)
    for ax, (_, label) in zip(axes, PANELS):
        ax.set_ylabel(label)
        style(ax)
    axes[-1].set_xlabel("file number across the dataset (each file ≈ 40 s; runs in name order)")
    for phase, color in PHASE_COLOR.items():
        axes[0].plot([], [], "o", color=color, label=phase)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}

    run_dirs = sorted(p for p in Path(args.data_root).glob(args.run_glob) if (p / "hit").exists())

    runs = []
    for rd in run_dirs:
        files = run_files(rd)
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        phase = phase_db.get(run_base.replace("_", ""))
        if phase is None:
            log.warning(f"{rd.name}: run {run_base} has no phase in {args.phases}; skipped")
            continue
        if run_base in {r["run"] for r in runs}:
            log.warning(f"{rd.name}: run {run_base} already read from another folder; skipped "
                        f"(use --run-glob to choose which copy)")
            continue
        log.info(f"{rd.name} ({phase}): {len(files)} files")
        per_file = [file_stats(f, args) for f in files]
        band = find_band(per_file, args.band_half_width)
        runs.append({"run": run_base, "phase": phase, "band": list(band),
                     "files": [summarize(d, band) for d in per_file]})

    plot(runs, out / "stability_vs_time.png")

    keys = ["median_psd", "median_npe_band", "median_prompt_pe", "median_slow_pe",
            "alpha_fraction", "pileup_fraction"]
    heads = ["PSD", "alpha PE", "prompt PE", "slow PE", "alpha frac", "pileup"]
    print("\nFirst and last few files of each run (first -> last)")
    print(f"{'run':<13}{'phase':<8}" + "".join(f"{h:>19}" for h in heads))
    for r in runs:
        f = r["files"]
        k = max(1, min(5, len(f) // 4))   # average a few files at each end

        def avg(key, part):
            v = [row[key] for row in part if np.isfinite(row[key])]
            return np.mean(v) if v else np.nan
        cells = []
        for key in keys:
            a, b = avg(key, f[:k]), avg(key, f[-k:])
            fmt = ".3f" if key in ("median_psd", "alpha_fraction", "pileup_fraction") else ".1f"
            cells.append(f"{a:{fmt}} -> {b:{fmt}}")
        print(f"{r['run']:<13}{r['phase']:<8}" + "".join(f"{c:>19}" for c in cells))

    with open(out / "stability_vs_time.json", "w") as fh:
        json.dump(runs, fh, indent=1, default=float)
    log.info(f"done: {out / 'stability_vs_time.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-glob", default="data_*",
                    help="Pattern for run folders under --data-root (default: data_*)")
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/stability")
    ap.add_argument("--bright", type=float, nargs=2, default=[15, 150], help="n_pe range of 'bright' events")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=50)
    ap.add_argument("--max-jump", type=float, default=3.0)
    ap.add_argument("--band-half-width", type=float, default=0.08)
    main(ap.parse_args())
