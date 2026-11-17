"""
Is the liquid's flat-topped prompt peak real, or the electronics compressing
very tall pulses?

For light that arrives almost all at once (liquid), the pulse's peak height
should grow in proportion to the number of photons. If the preamp/digitiser
compresses pulses near full scale, the peak height per photon falls as the
events get brighter, and the average pulse gets flatter and wider on top.
Solid has the same photons spread over microseconds, so its peaks stay far
below full scale: it is the control.

Peak height = maximum of the charge-window waveform (already baseline-
subtracted and positive-going, in ADC counts). For clipped events that
maximum is the distance from the baseline to the ADC floor, i.e. full scale,
so the clipped events show where the ceiling is.

Selection per event: no pileup (max_jump_pe < --max-jump), n_pe within
--n-min..--n-max, not saturated (lowest_adc >= floor + margin). No PSD cut,
so in the liquid all events are prompt-dominated anyway.

Outputs (in --output-dir):
  linearity.png   peak height vs n_pe and peak height per photon vs n_pe,
                  per window (liquid and the solid control), with the
                  clipping level marked
  slice_shapes.png  average pulse (unit area) of the first window in n_pe
                  slices, first 0.6 us: compression = lower, wider peaks
                  for the brighter slices
and a table: per window and slice, events, median peak per photon (and
relative to the dimmest slice), and the average pulse's FWHM.

Reads waveforms, so use a compute node for the full liquid range.

Usage (from the repo root):
    python analysis/liquid_linearity.py --output-dir results/advisor/13_liquid_linearity
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_raw_compass import parse_run_info  # noqa: E402
from saturation_check import find_runs  # noqa: E402

SAMPLE_NS = 2.0
WINDOW_COLOR = ["#eb6834", "#1baf7a", "#2a78d6", "#8a5cc2"]


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def parse_window(text):
    """'260520_1340:0-39' -> ('260520_1340', 0, 39)."""
    run, files = text.split(":")
    first, last = (int(v) for v in files.split("-"))
    return run, first, last


def file_stats(hit_path, dsp_path, args, slices, acc):
    """Peak height of every selected or saturated event in one file; adds the
    unit-charge waveforms of the selected events to the slice sums in acc."""
    t = pq.read_table(str(hit_path), columns=["n_pe", "lowest_adc", "max_jump_pe"])
    n = t.column("n_pe").to_numpy().astype(float)
    low = t.column("lowest_adc").to_numpy()
    jump = t.column("max_jump_pe").to_numpy()
    sat = low < args.sat_floor + args.sat_margin
    keep = (~sat) & (jump < args.max_jump) & (n >= args.n_min) & (n <= args.n_max)
    peaks = np.full(len(n), np.nan)
    offset = 0
    for batch in pq.ParquetFile(str(dsp_path)).iter_batches(batch_size=args.batch_size,
                                                            columns=["wf_charge_window"]):
        m = batch.num_rows
        k, s = keep[offset:offset + m], sat[offset:offset + m]
        if (k | s).any():
            vals = pc.struct_field(batch.column("wf_charge_window"), "values")
            flat = vals.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            W = flat.reshape(m, -1)                       # fixed-length window
            peaks[offset:offset + m] = W.max(axis=1)
            nn = n[offset:offset + m]
            for j, (lo, hi) in enumerate(slices):
                pick = k & (nn >= lo) & (nn < hi)
                if pick.any():
                    acc["sum"][j] = acc["sum"][j] + (W[pick] / nn[pick, None]).sum(axis=0)
                    acc["count"][j] += int(pick.sum())
        offset += m
    return n, peaks, keep, sat


def fwhm_ns(y):
    """Full width at half maximum of a pulse (linear interpolation on the edges)."""
    i = int(np.argmax(y))
    half = y[i] / 2
    left = i
    while left > 0 and y[left] > half:
        left -= 1
    right = i
    while right < len(y) - 1 and y[right] > half:
        right += 1
    xl = left + (half - y[left]) / (y[left + 1] - y[left]) if y[left + 1] != y[left] else left
    xr = right - 1 + (y[right - 1] - half) / (y[right - 1] - y[right]) if y[right - 1] != y[right] else right
    return (xr - xl) * SAMPLE_NS


def profile(n, v, edges):
    """Median and 16-84% band of v in bins of n."""
    mid, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (n >= a) & (n < b)
        if sel.sum() >= 20:
            mid.append(np.sqrt(a * b))
            q = np.percentile(v[sel], [16, 50, 84])
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    return np.array(mid), np.array(med), np.array(lo), np.array(hi)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args)
    slices = [tuple(float(v) for v in s.split("-")) for s in args.slices]

    results = []
    for w in args.windows:
        run, first, last = parse_window(w)
        rd, phase = runs[run]
        acc = {"sum": [0.0] * len(slices), "count": [0] * len(slices)}
        n_all, pk_all, keep_all, sat_all = [], [], [], []
        hits = [f for f in sorted((rd / "hit").glob("*.parquet"))
                if first <= parse_run_info(f.stem)[1] <= last]
        for i, hit_path in enumerate(hits):
            n, pk, keep, sat = file_stats(hit_path, rd / "dsp" / hit_path.name, args, slices, acc)
            n_all.append(n); pk_all.append(pk); keep_all.append(keep); sat_all.append(sat)
            print(f"  {w}: file {i + 1}/{len(hits)}", end="\r")
        print()
        n, pk = np.concatenate(n_all), np.concatenate(pk_all)
        keep, sat = np.concatenate(keep_all), np.concatenate(sat_all)
        ceiling = float(np.median(pk[sat])) if sat.sum() >= 5 else float("nan")
        results.append({"label": w, "phase": phase, "n": n[keep], "pk": pk[keep],
                        "ceiling": ceiling, "n_sat": int(sat.sum()), "acc": acc})

    # table
    print(f"\n{'window':<20}{'phase':<8}{'slice (PE)':>12}{'events':>10}{'peak/PE (ADC)':>15}"
          f"{'vs dimmest':>12}{'FWHM (ns)':>11}")
    for r in results:
        ref = None
        for (lo, hi), s, c in zip(slices, r["acc"]["sum"], r["acc"]["count"]):
            sel = (r["n"] >= lo) & (r["n"] < hi)
            if c < 20:
                continue
            ppe = float(np.median(r["pk"][sel] / r["n"][sel]))
            ref = ppe if ref is None else ref
            print(f"{r['label']:<20}{r['phase']:<8}{f'{lo:g}-{hi:g}':>12}{c:>10,}{ppe:>15.2f}"
                  f"{ppe / ref:>12.3f}{fwhm_ns(s / c):>11.1f}")
        print(f"{'':<20}{'':<8}{'clipped':>12}{r['n_sat']:>10,}"
              f"{'ceiling ' + format(r['ceiling'], '.0f') + ' ADC':>27}")

    # linearity plot
    edges = np.logspace(np.log10(args.n_min), np.log10(args.n_max), 31)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for r, c in zip(results, WINDOW_COLOR):
        mid, med, lo, hi = profile(r["n"], r["pk"], edges)
        ax1.plot(mid, med, "o-", color=c, markersize=4, label=f"{r['label']} ({r['phase']})")
        ax1.fill_between(mid, lo, hi, color=c, alpha=0.2)
        # straight line through the dim events: what a linear system would do
        dim = (r["n"] >= args.n_min) & (r["n"] < args.n_min + 15)
        if dim.sum() > 20:
            slope = float(np.median(r["pk"][dim] / r["n"][dim]))
            xx = np.array([args.n_min, args.n_max])
            ax1.plot(xx, slope * xx, ":", color=c, linewidth=1.2)
        mid, med, lo, hi = profile(r["n"], r["pk"] / r["n"], edges)
        ax2.plot(mid, med, "o-", color=c, markersize=4, label=r["label"])
        ax2.fill_between(mid, lo, hi, color=c, alpha=0.2)
        if np.isfinite(r["ceiling"]):
            ax1.axhline(r["ceiling"], color=c, linestyle="--", linewidth=1,
                        label=f"clipped pulses: {r['ceiling']:.0f} ADC")
    ax1.set_xlabel("n_pe")
    ax1.set_ylabel("peak height (ADC)")
    ax1.set_title("Peak height vs light (dotted: proportional to the dim events)")
    ax2.set_xlabel("n_pe")
    ax2.set_ylabel("peak height per photon (ADC/PE)")
    ax2.set_title("Linear electronics: flat for a fixed pulse shape")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "linearity.png", dpi=120)
    plt.close(fig)

    # slice shapes of each window, unit area, first 0.6 us around the peak
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        shown = [(sl, s / c) for sl, s, c in zip(slices, r["acc"]["sum"], r["acc"]["count"]) if c >= 20]
        for k, ((lo, hi), y) in enumerate(shown):
            y = y / y.sum()
            t = np.arange(len(y)) * SAMPLE_NS / 1000
            shade = 0.35 + 0.65 * k / max(len(shown) - 1, 1)
            ax.plot(t, y, color="#eb6834" if r["phase"] == "liquid" else "#1baf7a", alpha=shade,
                    linewidth=1.4, label=f"{lo:g}-{hi:g} PE (FWHM {fwhm_ns(y):.0f} ns)")
        ax.set_xlim(0, 0.6)
        ax.set_xlabel("time in charge window (μs)")
        ax.set_title(f"{r['label']} ({r['phase']}): average pulse by brightness")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0][0].set_ylabel("unit area")
    fig.tight_layout()
    fig.savefig(out / "slice_shapes.png", dpi=120)
    plt.close(fig)
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/liquid_linearity")
    ap.add_argument("--windows", nargs="+", default=["260520_1340:0-39", "260520_1800:70-79"],
                    help="RUN:FIRST-LAST file ranges; the first is the one tested, the rest are controls")
    ap.add_argument("--slices", nargs="+", default=["5-20", "20-35", "35-50", "50-65", "65-80"],
                    help="n_pe slices for the average shapes and the table")
    ap.add_argument("--n-min", type=float, default=5)
    ap.add_argument("--n-max", type=float, default=150)
    ap.add_argument("--max-jump", type=float, default=8)
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    ap.add_argument("--batch-size", type=int, default=2000)
    main(ap.parse_args())
