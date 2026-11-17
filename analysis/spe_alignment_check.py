"""
Is the single-PE response R too slow on its rising edge?

Every pulse-shape fit (old model, PEN, PEN + smear) leaves the same residual in
gas, liquid and solid: the data rise earlier and faster than the model, by up
to ~10% of the peak, then sit a little lower after the peak. A blur added to
the light can only make the model slower, so the fits push it to zero. That
points at R itself.

A likely cause: if R is the plain average of 1 PE events at the position
where each sits in its record, the trigger time of a 1 PE pulse jitters (a
single photon crosses the trigger threshold late and at a variable time), so
the average is the true 1 PE pulse blurred by that jitter. A 40-photon pulse
triggers on its first photons, with much less jitter, so it rises faster than
the blurred template allows.

This script, for each window:
  - takes 1 PE events (n_pe within --spe-range) and finds each pulse's
    onset (50% of its peak on the rising edge, interpolated between samples)
  - averages them two ways: as recorded, and shifted so all onsets line up
  - averages bright events (--data-range) as recorded, like the fits' data
  - compares the rise time (10-90%) and width of the three
and saves the aligned R (placed at the median onset, so the fits' start-time
convention is unchanged) for use in the fits.

Outputs (in --output-dir): spe_alignment.png, R_aligned_<run>.npy, and a table.

Usage (from the repo root, compute node):
    python analysis/spe_alignment_check.py --output-dir results/advisor/15_spe_alignment
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

SAMPLE_NS = 2.0


def onsets(W, frac=0.5, search=400):
    """Rising-edge crossing of frac x peak for each row of W (in samples,
    fractional), looking only at the first `search` samples."""
    sub = W[:, :search]
    sm = (sub[:, :-2] + sub[:, 1:-1] + sub[:, 2:]) / 3.0          # light smoothing
    ipk = np.argmax(sm, axis=1)
    out = np.full(len(W), np.nan)
    for i in range(len(W)):
        s, p = sm[i], ipk[i]
        level = frac * s[p]
        below = np.flatnonzero(s[:p] < level)
        if p == 0 or len(below) == 0:
            continue
        j = below[-1]
        out[i] = 1 + j + (level - s[j]) / (s[j + 1] - s[j])       # +1: smoothing offset
    return out


def edge_time(y, frac):
    """Fractional sample where y first rises through frac x max (before the peak)."""
    p = int(np.argmax(y))
    level = frac * y[p]
    below = np.flatnonzero(y[:p] < level)
    if len(below) == 0:
        return np.nan
    j = below[-1]
    return j + (level - y[j]) / (y[j + 1] - y[j])


def shape_numbers(y):
    """Rise 10-90% (ns), FWHM (ns), peak time after the 50% crossing (ns)."""
    p = int(np.argmax(y))
    t10, t50, t90 = (edge_time(y, f) for f in (0.1, 0.5, 0.9))
    after = np.flatnonzero(y[p:] < 0.5 * y[p])
    t50_fall = p + after[0] if len(after) else np.nan
    return (t90 - t10) * SAMPLE_NS, (t50_fall - t50) * SAMPLE_NS, (p - t50) * SAMPLE_NS


def collect(rd, first, last, args):
    """Waveforms of 1 PE events (up to --max-spe) and the summed bright events."""
    spe, data_sum, n_data = [], None, 0
    hits = [f for f in sorted((rd / "hit").glob("*.parquet"))
            if first <= parse_run_info(f.stem)[1] <= last]
    for hit_path in hits:
        t = pq.read_table(str(hit_path), columns=["n_pe", "lowest_adc", "max_jump_pe"])
        n = t.column("n_pe").to_numpy().astype(float)
        ok = (t.column("lowest_adc").to_numpy() >= args.sat_floor + args.sat_margin) & \
             (t.column("max_jump_pe").to_numpy() < args.max_jump)
        is_spe = ok & (n >= args.spe_range[0]) & (n <= args.spe_range[1])
        is_dat = ok & (n >= args.data_range[0]) & (n <= args.data_range[1])
        offset = 0
        for batch in pq.ParquetFile(str(rd / "dsp" / hit_path.name)).iter_batches(
                batch_size=args.batch_size, columns=["wf_charge_window"]):
            m = batch.num_rows
            a, b = is_spe[offset:offset + m], is_dat[offset:offset + m]
            if a.any() or b.any():
                vals = pc.struct_field(batch.column("wf_charge_window"), "values")
                W = vals.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(m, -1)
                if a.any() and sum(len(x) for x in spe) < args.max_spe:
                    spe.append(W[a])
                if b.any():
                    s = (W[b] / n[offset:offset + m][b, None]).sum(axis=0)
                    data_sum = s if data_sum is None else data_sum + s
                    n_data += int(b.sum())
            offset += m
    return np.concatenate(spe)[:args.max_spe], data_sum / max(n_data, 1), n_data


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args)
    fig, axes = plt.subplots(2, len(args.windows), figsize=(7 * len(args.windows), 8), squeeze=False)
    print(f"\n{'window':<20}{'curve':<22}{'events':>9}{'rise 10-90 (ns)':>17}{'FWHM (ns)':>11}"
          f"{'peak after 50% (ns)':>21}")
    for j, w in enumerate(args.windows):
        run, first, last = parse_window(w)
        rd, phase = runs[run]
        spe, data, n_data = collect(rd, first, last, args)
        t_on = onsets(spe)
        good = np.isfinite(t_on)
        spe, t_on = spe[good], t_on[good]
        ref = float(np.median(t_on))
        x = np.arange(spe.shape[1], dtype=float)
        plain = spe.mean(axis=0)
        aligned = np.zeros(spe.shape[1])
        for wf, t in zip(spe, t_on):
            aligned += np.interp(x + (t - ref), x, wf)
        aligned /= len(spe)
        np.save(out / f"R_aligned_{run}.npy", aligned / aligned.sum())

        spread = np.percentile(t_on, [16, 84]) * SAMPLE_NS
        print(f"{w:<20}{'1 PE onset jitter':<22}{len(t_on):>9,}"
              f"   std {t_on.std() * SAMPLE_NS:.1f} ns, 16-84% {spread[1] - spread[0]:.1f} ns")
        curves = [("R as recorded", plain, len(spe)), ("R aligned", aligned, len(spe)),
                  (f"data {args.data_range[0]:g}-{args.data_range[1]:g} PE", data, n_data)]
        ax = axes[0][j]
        for (label, y, cnt), c in zip(curves, ["#999999", "#2a78d6", "#eb6834"]):
            r, fw, pk = shape_numbers(y)
            print(f"{'':<20}{label:<22}{cnt:>9,}{r:>17.1f}{fw:>11.1f}{pk:>21.1f}")
            t50 = edge_time(y, 0.5)
            ax.plot((x - t50) * SAMPLE_NS, y / y.max(), color=c, linewidth=1.5,
                    label=f"{label}: rise {r:.0f} ns")
        ax.set_xlim(-60, 400)
        ax.set_xlabel("time from 50% crossing (ns)")
        ax.set_ylabel("peak = 1")
        ax.set_title(f"{w} ({phase})")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
        ax = axes[1][j]
        ax.hist((t_on - ref) * SAMPLE_NS, bins=np.arange(-60, 61, 2), color="#2a78d6", alpha=0.7)
        ax.set_xlabel("1 PE onset relative to median (ns)")
        ax.set_ylabel("events")
        ax.set_title("Where single photons sit in their records")
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "spe_alignment.png", dpi=120)
    plt.close(fig)
    print(f"\nplots and aligned R in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/spe_alignment")
    ap.add_argument("--windows", nargs="+", default=["260520_1340:0-4", "260520_1800:70-74", "260520_1239:20-24"])
    ap.add_argument("--spe-range", nargs=2, type=float, default=[0.8, 1.2])
    ap.add_argument("--data-range", nargs=2, type=float, default=[20, 50])
    ap.add_argument("--max-spe", type=int, default=50000)
    ap.add_argument("--max-jump", type=float, default=8)
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    ap.add_argument("--batch-size", type=int, default=2000)
    main(ap.parse_args())
