"""
Where do multi-photon pulses start in their records?

Single photons sit at the same place in every record (16-84% spread ~1.6 ns),
so the 1 PE response R is sharp (10-90% rise ~14 ns). Yet the average of
20-50 PE events begins to rise ~25 ns before its 50% point with a slow, curved
"toe" -- the shape you get when the pulses in the average start at different
times. If light physics were the only thing at work, every event's first
photons would arrive at the same moment after the alpha, and the event's
start would sit at a fixed place in the record.

This script measures each event's own start (the 10% and 50% crossings of its
rising edge) and asks:
  - how wide is the spread of starts among events of similar brightness?
    (a spread of ~10-20 ns would explain the toe and the rising-edge residual)
  - does the start move with brightness? (a trend with n_pe = trigger time
    walk, an electronics effect, not light)

Starts are given relative to the median start of single photons in the same
files, measured the same way.

Outputs (in --output-dir): data_onsets.png and a table.

Usage (from the repo root, compute node):
    python analysis/data_onset_check.py --output-dir results/advisor/16_data_onsets
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
from liquid_linearity import parse_window, style  # noqa: E402
from spe_alignment_check import onsets  # noqa: E402

SAMPLE_NS = 2.0


def collect(rd, first, last, args):
    """n_pe and the 10% / 50% start (samples) of every selected event, plus
    the same for single photons."""
    n_out, s10, s50, spe10, spe50 = [], [], [], [], []
    hits = [f for f in sorted((rd / "hit").glob("*.parquet"))
            if first <= parse_run_info(f.stem)[1] <= last]
    for hit_path in hits:
        t = pq.read_table(str(hit_path), columns=["n_pe", "lowest_adc", "max_jump_pe"])
        n = t.column("n_pe").to_numpy().astype(float)
        ok = (t.column("lowest_adc").to_numpy() >= args.sat_floor + args.sat_margin) & \
             (t.column("max_jump_pe").to_numpy() < args.max_jump)
        is_spe = ok & (n >= 0.8) & (n <= 1.2)
        is_dat = ok & (n >= args.n_min) & (n <= args.n_max)
        offset = 0
        for batch in pq.ParquetFile(str(rd / "dsp" / hit_path.name)).iter_batches(
                batch_size=args.batch_size, columns=["wf_charge_window"]):
            m = batch.num_rows
            a, b = is_spe[offset:offset + m], is_dat[offset:offset + m]
            if a.any() or b.any():
                vals = pc.struct_field(batch.column("wf_charge_window"), "values")
                W = vals.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(m, -1)
                W = W[:, :args.search]
                if b.any():
                    n_out.append(n[offset:offset + m][b])
                    s10.append(onsets(W[b], 0.1, args.search))
                    s50.append(onsets(W[b], 0.5, args.search))
                if a.any() and sum(len(x) for x in spe10) < 20000:
                    spe10.append(onsets(W[a], 0.1, args.search))
                    spe50.append(onsets(W[a], 0.5, args.search))
            offset += m
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return cat(n_out), cat(s10), cat(s50), cat(spe10), cat(spe50)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args)
    slices = [tuple(float(v) for v in s.split("-")) for s in args.slices]
    fig, axes = plt.subplots(2, len(args.windows), figsize=(7 * len(args.windows), 9), squeeze=False)
    print(f"\n{'window':<20}{'n_pe':>10}{'events':>10}{'10% start (ns)':>16}{'spread 16-84':>14}"
          f"{'50% start (ns)':>16}{'spread 16-84':>14}")
    for j, w in enumerate(args.windows):
        run, first, last = parse_window(w)
        rd, phase = runs[run]
        n, s10, s50, p10, p50 = collect(rd, first, last, args)
        ref10, ref50 = np.nanmedian(p10), np.nanmedian(p50)
        d10, d50 = (s10 - ref10) * SAMPLE_NS, (s50 - ref50) * SAMPLE_NS
        q = lambda v: np.nanpercentile(v, [16, 84])
        r10, r50 = q((p10 - ref10) * SAMPLE_NS), q((p50 - ref50) * SAMPLE_NS)
        print(f"{w:<20}{'1 PE':>10}{len(p10):>10,}{0:>16.1f}{r10[1] - r10[0]:>14.1f}"
              f"{0:>16.1f}{r50[1] - r50[0]:>14.1f}")
        for lo, hi in slices:
            sel = (n >= lo) & (n < hi)
            if sel.sum() < 50:
                continue
            a, b = q(d10[sel]), q(d50[sel])
            print(f"{'':<20}{f'{lo:g}-{hi:g}':>10}{sel.sum():>10,}{np.nanmedian(d10[sel]):>16.1f}"
                  f"{a[1] - a[0]:>14.1f}{np.nanmedian(d50[sel]):>16.1f}{b[1] - b[0]:>14.1f}")

        ax = axes[0][j]
        good = np.isfinite(d10)
        ax.hist2d(n[good], d10[good], bins=[np.logspace(np.log10(args.n_min), np.log10(args.n_max), 40),
                                            np.arange(-80, 41, 2)], cmap="Blues", cmin=1)
        edges = np.logspace(np.log10(args.n_min), np.log10(args.n_max), 21)
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = good & (n >= lo) & (n < hi)
            if sel.sum() >= 50:
                mids.append(np.sqrt(lo * hi)); meds.append(np.median(d10[sel]))
        ax.plot(mids, meds, "o-", color="#eb6834", markersize=4, label="median")
        ax.axhline(0, color="#222222", linestyle="--", linewidth=1, label="single photons")
        ax.set_xscale("log")
        ax.set_xlabel("n_pe")
        ax.set_ylabel("10% start relative to single photons (ns)")
        ax.set_title(f"{w} ({phase}): where events start")
        ax.legend(frameon=False, fontsize=9)
        style(ax)

        ax = axes[1][j]
        bins = np.arange(-80, 41, 1)
        ax.hist((p10 - ref10) * SAMPLE_NS, bins=bins, histtype="step", color="#555555",
                density=True, linewidth=1.4, label="single photons")
        for k, (lo, hi) in enumerate(slices):
            sel = good & (n >= lo) & (n < hi)
            if sel.sum() >= 50:
                ax.hist(d10[sel], bins=bins, histtype="step", density=True, linewidth=1.4,
                        color="#2a78d6", alpha=0.35 + 0.65 * k / max(len(slices) - 1, 1),
                        label=f"{lo:g}-{hi:g} PE")
        ax.set_xlabel("10% start relative to single photons (ns)")
        ax.set_ylabel("fraction of events per ns")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "data_onsets.png", dpi=120)
    plt.close(fig)
    print(f"\nplot in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/data_onsets")
    ap.add_argument("--windows", nargs="+", default=["260520_1340:0-4", "260520_1800:70-74", "260520_1239:20-24"])
    ap.add_argument("--slices", nargs="+", default=["5-20", "20-35", "35-50", "50-65"])
    ap.add_argument("--n-min", type=float, default=3)
    ap.add_argument("--n-max", type=float, default=80)
    ap.add_argument("--search", type=int, default=400, help="Samples searched for the start and peak")
    ap.add_argument("--max-jump", type=float, default=8)
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    ap.add_argument("--batch-size", type=int, default=2000)
    main(ap.parse_args())
