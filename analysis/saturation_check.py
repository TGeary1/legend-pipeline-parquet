"""
Events beyond the digitiser's dynamic range: what the saturation cut removes.

A pulse bigger than the ADC range is clipped at the bottom of the range
(the "floor", ~7,067 ADC here; pulses go negative), so it shows a flat,
trapezoidal top. Its charge is then underestimated and its PSD is wrong.
The saturation cut removes events whose lowest sample comes within
--sat-margin ADC of the floor. Very bright events like these are most
likely cosmic-ray muons (or alpha-on-alpha pileup) crossing the argon.

Outputs (in --output-dir):
  saturation_spectrum.png  n_pe spectrum of all vs saturated events, per
                           phase, and the distribution of lowest_adc near
                           the floor (where the cut sits)
  saturation_gallery.png   charge-window waveforms of saturated events,
                           next to the brightest events that pass the cut

The spectrum reads the hit tier only; the gallery also reads a few
waveforms from one dsp file. Fine on a login node.

Usage (from the repo root):
    python analysis/saturation_check.py --output-dir results/advisor/09_saturation
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}
SAMPLE_NS = 2.0


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def find_runs(args):
    """{run_base: (run folder, phase)} for runs with a hit tier."""
    with open(args.phases) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    runs = {}
    for rd in sorted(Path(args.data_root).glob(args.run_glob)):
        files = list((rd / "hit").glob("*.parquet"))
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        phase = phase_db.get(run_base.replace("_", ""))
        if phase and run_base not in runs:
            runs[run_base] = (rd, phase)
    return runs


def load_phase(run_dirs):
    """n_pe and lowest_adc of every event in these runs."""
    n_all, low_all = [], []
    for rd in run_dirs:
        for f in sorted((rd / "hit").glob("*.parquet")):
            t = pq.read_table(str(f), columns=["n_pe", "lowest_adc"])
            n_all.append(t.column("n_pe").to_numpy())
            low_all.append(t.column("lowest_adc").to_numpy())
    return np.concatenate(n_all), np.concatenate(low_all)


def waveforms(dsp_path, rows, batch_size=5000):
    """Charge-window waveforms of the given row numbers of one dsp file."""
    want = set(int(r) for r in rows)
    out, offset = {}, 0
    for batch in pq.ParquetFile(str(dsp_path)).iter_batches(batch_size=batch_size,
                                                            columns=["wf_charge_window"]):
        n = batch.num_rows
        hit = [r for r in want if offset <= r < offset + n]
        if hit:
            vals = pc.struct_field(batch.column("wf_charge_window"), "values")
            for r in hit:
                out[r] = np.asarray(vals[r - offset].as_py(), dtype=float)
        offset += n
        if len(out) == len(want):
            break
    return [out[r] for r in rows if r in out]


def plot_spectrum(by_phase, args, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    bins = np.logspace(0, np.log10(3000), 121)
    for phase, (n, low) in by_phase.items():
        sat = low < args.sat_floor + args.sat_margin
        c = PHASE_COLOR.get(phase, "#777777")
        ax1.hist(n, bins=bins, histtype="step", color=c, linewidth=1.4, label=f"{phase}, all")
        ax1.hist(n[sat], bins=bins, histtype="stepfilled", color=c, alpha=0.35,
                 label=f"{phase}, saturated ({sat.mean():.3%})")
        ax2.hist(low, bins=np.arange(args.sat_floor - 20, args.sat_floor + 400, 5), histtype="step",
                 color=c, linewidth=1.4, label=phase)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("n_pe (photoelectrons; underestimated for saturated events)")
    ax1.set_ylabel("events")
    ax1.set_title("Energy spectrum: all events vs saturated")
    ax1.legend(frameon=False, fontsize=9)
    ax2.axvline(args.sat_floor + args.sat_margin, color="#222222", linestyle="--", linewidth=1.2,
                label=f"cut: floor + {args.sat_margin:g}")
    ax2.set_yscale("log")
    ax2.set_xlabel("lowest ADC value in the record")
    ax2.set_ylabel("events")
    ax2.set_title("Pulse depth near the ADC floor")
    ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gallery(sat_wfs, bright_wfs, label, path):
    ncol = 4
    nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 7), sharex=True)
    for i, ax in enumerate(axes.flat):
        group, k = (sat_wfs, i) if i < ncol else (bright_wfs, i - ncol)
        if k < len(group):
            w = group[k]
            t = np.arange(len(w)) * SAMPLE_NS / 1000
            ax.plot(t, w, color="#c0392b" if i < ncol else "#2a78d6", linewidth=0.8)
            ax.set_xlim(0, 2.5)
        ax.set_title(("saturated" if i < ncol else "brightest unsaturated") + f" #{k + 1}", fontsize=11)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time in charge window (μs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("charge-window waveform (ADC)")
    fig.suptitle(f"{label}: clipped pulses have flat (trapezoidal) tops")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args)

    by_phase = {}
    print(f"\n{'run':<14}{'phase':<8}{'events':>12}{'saturated':>11}{'median n_pe, sat.':>19}")
    for run_base, (rd, phase) in runs.items():
        n, low = load_phase([rd])
        sat = low < args.sat_floor + args.sat_margin
        med = np.median(n[sat]) if sat.any() else float("nan")
        print(f"{run_base:<14}{phase:<8}{len(n):>12,}{sat.mean():>11.3%}{med:>19.0f}")
        prev = by_phase.get(phase)
        by_phase[phase] = (n, low) if prev is None else (np.r_[prev[0], n], np.r_[prev[1], low])
    plot_spectrum(by_phase, args, out / "saturation_spectrum.png")

    # gallery: first file of the chosen run that has enough saturated events
    rd, phase = runs[args.gallery_run]
    for hit_path in sorted((rd / "hit").glob("*.parquet"), key=lambda f: parse_run_info(f.stem)[1]):
        t = pq.read_table(str(hit_path), columns=["n_pe", "lowest_adc"])
        n, low = t.column("n_pe").to_numpy(), t.column("lowest_adc").to_numpy()
        sat_rows = np.flatnonzero(low < args.sat_floor + args.sat_margin)
        if len(sat_rows) >= 4:
            ok = np.flatnonzero(low >= args.sat_floor + args.sat_margin)
            bright_rows = ok[np.argsort(n[ok])[-4:]][::-1]
            dsp_path = rd / "dsp" / hit_path.name
            plot_gallery(waveforms(dsp_path, sat_rows[:4]), waveforms(dsp_path, bright_rows),
                         f"{args.gallery_run} ({phase}), {hit_path.name}", out / "saturation_gallery.png")
            break
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--run-glob", default="data_2*")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/saturation")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=180)
    ap.add_argument("--gallery-run", default="260520_1340", help="Run whose waveforms go in the gallery")
    main(ap.parse_args())
