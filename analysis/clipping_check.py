"""
Find clipped (flat-topped) pulses directly from their shape, and compare
with the current saturation cut on lowest_adc.

A clipped pulse sits at its maximum for many samples: the top of the
trapezoid. An unclipped pulse is only at its maximum for a sample or two,
because it starts decaying straight after the peak. So the number of
samples within --tol ADC of the event's own maximum ("flat-top length")
separates the two, whatever level the clipping happened at.

For a few files per run, reads the charge-window waveforms (in batches)
and the hit tier, and for every bright event (peak above --min-peak ADC)
records its flat-top length and lowest_adc.

Outputs (in --output-dir):
  clipping_check.png  flat-top length vs lowest_adc (with the current cut)
                      and vs n_pe, all runs together

Usage (from the repo root; a login node is enough for 2 files per run):
    python analysis/clipping_check.py --output-dir results/advisor/09_saturation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def flat_top(w, tol):
    """Samples within tol ADC of each waveform's own maximum, and the maximum."""
    peak = w.max(axis=1)
    return (w >= (peak - tol)[:, None]).sum(axis=1), peak


def file_flat_tops(hit_path, dsp_path, args):
    """Flat-top length, peak, lowest_adc and n_pe of the bright events of one file."""
    t = pq.read_table(str(hit_path), columns=["n_pe", "lowest_adc"])
    n_pe, low = t.column("n_pe").to_numpy(), t.column("lowest_adc").to_numpy()
    flats, peaks = [], []
    for batch in pq.ParquetFile(str(dsp_path)).iter_batches(batch_size=5000, columns=["wf_charge_window"]):
        vals = pc.struct_field(batch.column("wf_charge_window"), "values")
        w = np.stack(vals.to_numpy(zero_copy_only=False)).astype(np.float32)
        f, p = flat_top(w, args.tol)
        flats.append(f)
        peaks.append(p)
    flat, peak = np.concatenate(flats), np.concatenate(peaks)
    bright = peak > args.min_peak
    return flat[bright], peak[bright], low[bright], n_pe[bright]


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cut = args.sat_floor + args.sat_margin
    parts = []
    print(f"\n{'run':<14}{'bright events':>15}{'flat >= N':>11}{'flat, pass cut':>16}"
          f"{'not flat, fail cut':>20}{'lowest_adc of flat (5-95%)':>28}")
    for rd in sorted(Path(args.data_root).glob(args.run_glob)):
        files = sorted((rd / "hit").glob("*.parquet"), key=lambda f: parse_run_info(f.stem)[1])
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        picks = files[:: max(1, len(files) // args.files_per_run)][:args.files_per_run]
        res = [file_flat_tops(f, rd / "dsp" / f.name, args) for f in picks]
        flat, peak, low, n = (np.concatenate(x) for x in zip(*res))
        is_flat = flat >= args.min_flat
        fails = low < cut
        rng = (f"{np.percentile(low[is_flat], 5):.0f}-{np.percentile(low[is_flat], 95):.0f}"
               if is_flat.any() else "-")
        print(f"{run_base:<14}{len(flat):>15,}{is_flat.sum():>11,}{(is_flat & ~fails).sum():>16,}"
              f"{(~is_flat & fails).sum():>20,}{rng:>28}")
        parts.append((flat, low, n))

    flat, low, n = (np.concatenate(x) for x in zip(*parts))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    fbins = np.arange(0, 400, 4)
    h, xe, ye = np.histogram2d(low, flat, bins=(np.arange(7000, 8500, 10), fbins))
    m = ax1.pcolormesh(xe, ye, np.where(h.T > 0, h.T, np.nan), norm=LogNorm(), cmap="viridis")
    ax1.axvline(cut, color="#222222", linestyle="--", linewidth=1.2, label=f"current cut ({cut:g})")
    ax1.axhline(args.min_flat, color="#c0392b", linestyle=":", linewidth=1.5,
                label=f"flat top ≥ {args.min_flat} samples")
    ax1.set_xlabel("lowest ADC value in the record")
    ax1.set_ylabel(f"flat-top length (samples within {args.tol:g} ADC of the peak)")
    ax1.set_title("Clipping vs the current lowest_adc cut")
    ax1.legend(frameon=False, fontsize=9)
    fig.colorbar(m, ax=ax1, label="events")
    h, xe, ye = np.histogram2d(n, flat, bins=(np.logspace(0.5, 3, 80), fbins))
    m = ax2.pcolormesh(xe, ye, np.where(h.T > 0, h.T, np.nan), norm=LogNorm(), cmap="viridis")
    ax2.set_xscale("log")
    ax2.axhline(args.min_flat, color="#c0392b", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("n_pe (photoelectrons; underestimated when clipped)")
    ax2.set_ylabel("flat-top length (samples)")
    ax2.set_title("Clipping vs energy")
    fig.colorbar(m, ax=ax2, label="events")
    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "clipping_check.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out / 'clipping_check.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--output-dir", default="results/saturation")
    ap.add_argument("--files-per-run", type=int, default=2, help="Files read per run, spread across the run")
    ap.add_argument("--min-peak", type=float, default=3000, help="Only events whose waveform peak exceeds this (ADC)")
    ap.add_argument("--tol", type=float, default=15, help="ADC below the peak that still counts as 'at the top'")
    ap.add_argument("--min-flat", type=int, default=10, help="Flat-top length (samples) that marks a clipped pulse")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    main(ap.parse_args())
