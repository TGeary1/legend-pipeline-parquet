"""
Standard calibration and data-quality plots for one run, in the same spirit
as the per-dataset section of the previous analysis (102025_data.py), so every
run in this study gets an identical, comparable set of figures.

Plots produced (all in <output-dir>/<label>/):
  1. <label>_baseline.png           baseline level and baseline noise
  2. <label>_charge_fit.png         charge histogram with the 1..N PE peaks
                                    fitted, plus peak position vs PE number
                                    (the gain line)
  3. <label>_npe_check.png          calibrated n_pe near 0-10 PE: the peaks
                                    should sit on the integers
  4. <label>_energy_spectrum.png    n_pe spectrum (log and linear)
  5. <label>_pulse_timing.png       pulse arrival time and pulse height
  6. <label>_spe_pulse.png          average 1 PE pulse (linear and log)
  7. <label>_event_rate.png         recorded event rate vs time in the run
  plus <label>_summary.json with the numbers behind them.

What each plot is for:
  - Baseline: events whose pre-pulse baseline is shifted or noisy (often the
    tail of an earlier pulse) show up as tails here. Nothing is cut yet;
    this is to see whether a cut is needed.
  - Charge fit: an independent re-fit of the photoelectron peaks, drawn so
    you can see the fit, and compared to the gain stored in
    config/spe_calibration.yaml. The two should agree closely (small
    differences come from binning). The YAML value is the one the pipeline
    uses.
  - n_pe check: after calibration, the 1, 2, 3... PE peaks must land on
    1, 2, 3... If they drift away from the integers, the gain is wrong.
  - Energy spectrum: the main physics distribution for the run.
  - Pulse timing: the trigger should put every pulse at the same place in
    the record. A wide or double-peaked arrival-time distribution means
    pileup or trigger problems.
  - 1 PE pulse: the single-photon response that deconvolution divides out.
  - Event rate: a stable run has a flat recorded rate. Drops or jumps point
    to DAQ problems, source movement, or changing conditions.

Reads only the columns it needs. Waveform-based plots (pulse timing, 1 PE
pulse) use a sample of events from the first few files, so memory stays
small; the rest use every hit file. The event-rate plot reads only the
`timestamp` column of the raw tier.

Usage (from the repo root):
    python analysis/calibration_plots.py \\
        --run-dir /global/cfs/cdirs/m2676/users/tgeary/legend-pipeline-parquet/data_260520_1340 \\
        --output-dir results/calibration

A run with no entry in config/spe_calibration.yaml still gets the baseline,
timing and rate plots; the calibration-dependent plots are skipped with a
message. Pass --peak PE:LO:HI (same format as calibration/spe_fit.py) to fit specific
windows instead of windows centred on the stored gain.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Run the script as `python analysis/calibration_plots.py`: Python then puts
# analysis/ (not the repo root) on the import path, so add the root to reach
# build_raw_compass.parse_run_info, the same run-name parser the pipeline uses.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_NS = 2.0          # DT5730 native 500 MS/s (PIPELINE_OVERVIEW_PARQUET.md §7a)
TIMESTAMP_UNIT_S = 1e-12  # CoMPASS timestamps are in picoseconds

# One ink colour for single-series plots, one accent for fits/markers.
DATA_COLOR = "#2a6f97"
FIT_COLOR = "#c8553d"
REF_COLOR = "#555555"


def style(ax):
    """Recessive grid, no top/right box, so the data carries the plot."""
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_hit(hit_dir, n_files=None):
    """Concatenate the scalar hit-tier columns this script needs. n_pe is
    only present in runs that were calibrated when their hit tier was made."""
    files = sorted(Path(hit_dir).glob("*.parquet"))
    if n_files:
        files = files[:n_files]
    if not files:
        raise FileNotFoundError(f"no hit files in {hit_dir}")

    wanted = ["charge_total", "n_pe", "bl", "bl_sig", "wf_amplitude", "psd_param"]
    cols = {c: [] for c in wanted}
    for f in files:
        names = pq.ParquetFile(str(f)).schema_arrow.names
        present = [c for c in wanted if c in names]
        t = pq.read_table(str(f), columns=present)
        for c in wanted:
            if c in present:
                cols[c].append(t.column(c).to_numpy())
    data = {c: np.concatenate(v) for c, v in cols.items() if v and len(v) == len(files)}
    log.info(f"hit: {len(files)} files, {len(data['charge_total']):,} events, "
             f"columns {sorted(data)}")
    return data, files


def sample_waveforms(dsp_dir, hit_files, n_files, n_events, batch_size=5000):
    """Walk the first `n_files` dsp files in batches and keep only what the
    timing and 1 PE plots need: arrival time, n_pe, and a running sum of
    1 PE waveforms. Never holds more than one batch of waveforms at once."""
    arrival_ns, arrival_npe = [], []
    spe_sum, n_spe = None, 0
    for hit_f in hit_files[:n_files]:
        dsp_f = Path(dsp_dir) / hit_f.name
        if not dsp_f.exists():
            log.warning(f"no dsp file for {hit_f.name}; skipping in waveform sample")
            continue
        hit_names = pq.ParquetFile(str(hit_f)).schema_arrow.names
        if "n_pe" not in hit_names:
            log.warning("hit files have no n_pe; waveform sample skipped")
            return None
        n_pe_all = pq.read_table(str(hit_f), columns=["n_pe"]).column("n_pe").to_numpy()

        pf = pq.ParquetFile(str(dsp_f))
        if pf.metadata.num_rows != len(n_pe_all):
            raise ValueError(f"row mismatch: {dsp_f.name} dsp {pf.metadata.num_rows} "
                             f"vs hit {len(n_pe_all)}")
        offset = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=["wf_charge_window"]):
            vals = pc.struct_field(batch.column("wf_charge_window"), "values")
            wfs = np.stack(vals.to_numpy(zero_copy_only=False)).astype(np.float32)
            n_pe = n_pe_all[offset:offset + len(wfs)]
            offset += len(wfs)

            # arrival time: first sample reaching 50% of the pulse maximum
            # (the same definition as the previous analysis's time_point(wf, 50))
            lit = n_pe > 0.5
            w = wfs[lit]
            above = w >= 0.5 * w.max(axis=1, keepdims=True)
            arrival_ns.append(above.argmax(axis=1) * SAMPLE_NS)
            arrival_npe.append(n_pe[lit])

            is_1pe = (n_pe > 0.85) & (n_pe < 1.15)
            s = wfs[is_1pe].sum(axis=0, dtype=np.float64)
            spe_sum = s if spe_sum is None else spe_sum + s
            n_spe += int(is_1pe.sum())
            if offset >= n_events:
                break
        log.info(f"waveform sample: {dsp_f.name}, {offset:,} events read")

    if not arrival_ns:
        return None
    return {
        "arrival_ns": np.concatenate(arrival_ns),
        "arrival_npe": np.concatenate(arrival_npe),
        "spe_avg": spe_sum / max(n_spe, 1),
        "n_spe": n_spe,
    }


def load_timestamps(raw_dir):
    """Read only the timestamp column of every raw file."""
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        return None
    ts = [pq.read_table(str(f), columns=["timestamp"]).column("timestamp").to_numpy()
          for f in files]
    # check the clock keeps running across files rather than restarting
    restarts = sum(1 for a, b in zip(ts[:-1], ts[1:]) if len(a) and len(b) and b[0] < a[-1])
    if restarts:
        log.warning(f"timestamps restart in {restarts} file boundaries; "
                    f"the rate plot will be wrong. Tell Claude.")
    t = np.sort(np.concatenate(ts).astype(np.float64))
    return (t - t[0]) * TIMESTAMP_UNIT_S


def lookup(db, run_base, yaml_path):
    """Find run_base in a YAML dict. An unquoted key like 260520_1340 is read
    by YAML as the integer 2605201340, so also try that form and warn."""
    if run_base in db:
        return db[run_base]
    as_int = run_base.replace("_", "")
    for k, v in db.items():
        if str(k) == as_int:
            log.warning(f"{yaml_path}: key for {run_base} is unquoted and was read as a number; "
                        f"quote it ('{run_base}':) so the pipeline finds it too")
            return v
    return None


def load_calibration(yaml_path, run_base):
    with open(yaml_path) as f:
        db = yaml.safe_load(f) or {}
    return lookup(db, run_base, yaml_path)


def load_phase(yaml_path, run_base):
    try:
        with open(yaml_path) as f:
            db = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    return lookup(db, run_base, yaml_path)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_peaks(charge, windows, bin_width):
    """Fit a Gaussian inside each (pe, lo, hi) window of the charge histogram,
    then a straight line through the peak means vs PE number. The slope of
    that line is the gain (charge per photoelectron); the intercept is the
    charge offset (pedestal)."""
    lo_all = min(w[1] for w in windows) - 2 * bin_width
    hi_all = max(w[2] for w in windows) + 2 * bin_width
    edges = np.arange(lo_all, hi_all + bin_width, bin_width)
    counts, edges = np.histogram(charge, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peaks = []
    for pe, lo, hi in windows:
        m = (centers >= lo) & (centers <= hi)
        x, y = centers[m], counts[m]
        if y.sum() < 20:
            log.warning(f"{pe} PE window [{lo:.0f}, {hi:.0f}] has too few events; skipped")
            continue
        p0 = [y.max(), x[np.argmax(y)], (hi - lo) / 4]
        err = np.sqrt(np.maximum(y, 1))   # Poisson error on each bin
        try:
            p, cov = curve_fit(gaussian, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=10000)
        except RuntimeError:
            log.warning(f"{pe} PE Gaussian fit did not converge; skipped")
            continue
        peaks.append({"pe": pe, "lo": lo, "hi": hi, "amp": p[0], "mu": p[1],
                      "mu_err": float(np.sqrt(cov[1, 1])), "sigma": abs(p[2])})

    if len(peaks) < 2:
        return None
    pe = np.array([p["pe"] for p in peaks], dtype=float)
    mu = np.array([p["mu"] for p in peaks])
    mu_err = np.array([p["mu_err"] for p in peaks])
    (gain, ped), cov = curve_fit(lambda k, g, b: g * k + b, pe, mu, sigma=mu_err, absolute_sigma=True)
    return {"peaks": peaks, "gain": float(gain), "gain_err": float(np.sqrt(cov[0, 0])),
            "pedestal": float(ped), "pedestal_err": float(np.sqrt(cov[1, 1]))}


def windows_from_gain(gain, pedestal, n_peaks, half_width=0.25):
    """Windows centred on where the stored calibration says each peak is."""
    return [(k, pedestal + k * gain - half_width * gain, pedestal + k * gain + half_width * gain)
            for k in range(1, n_peaks + 1)]


def parse_peak_arg(s):
    pe, lo, hi = s.split(":")
    return int(pe), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_baseline(d, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary = {}
    for ax, col, xlabel in ((axes[0], "bl", "baseline level [ADC]"),
                            (axes[1], "bl_sig", "baseline noise (std. dev.) [ADC]")):
        v = d[col][np.isfinite(d[col])]
        p01, p50, p99 = np.percentile(v, [0.1, 50, 99.9])
        span = max(p99 - p01, 1e-9)
        lo, hi = p01 - 0.5 * span, p99 + 0.5 * span
        ax.hist(np.clip(v, lo, hi), bins=200, range=(lo, hi), histtype="step", color=DATA_COLOR)
        ax.axvline(p50, color=REF_COLOR, linestyle="--", linewidth=1, label=f"median {p50:.1f}")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("events")
        ax.legend(frameon=False)
        style(ax)
        frac_out = float(np.mean((v < p01 - 0.5 * span) | (v > p99 + 0.5 * span)))
        summary[col] = {"median": float(p50), "p0.1": float(p01), "p99.9": float(p99),
                        "frac_beyond_plot": frac_out}
    fig.suptitle(f"{title}: baseline quality (values beyond the range are piled in the end bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


def plot_charge_fit(charge, fit, cal, path, title, n_peaks_shown):
    gain_ref = cal["gain"] if cal else fit["gain"]
    ped_ref = cal.get("pedestal", 0.0) if cal else fit["pedestal"]
    bin_width = gain_ref / 50
    hi = ped_ref + (n_peaks_shown + 1.5) * gain_ref
    lo = min(0.0, ped_ref - 0.5 * gain_ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    ax1.hist(charge[(charge > lo) & (charge < hi)], bins=np.arange(lo, hi, bin_width),
             histtype="step", color=DATA_COLOR, label="data")
    for i, p in enumerate(fit["peaks"]):
        x = np.linspace(p["lo"], p["hi"], 200)
        ax1.plot(x, gaussian(x, p["amp"], p["mu"], p["sigma"]), color=FIT_COLOR, linewidth=2,
                 label="Gaussian fits" if i == 0 else None)
        ax1.annotate(f"{p['pe']} PE", (p["mu"], p["amp"]), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_xlabel("charge_total [charge units]")
    ax1.set_ylabel(f"events per {bin_width:.0f}")
    ax1.legend(frameon=False)
    style(ax1)

    pe = np.array([p["pe"] for p in fit["peaks"]])
    mu = np.array([p["mu"] for p in fit["peaks"]])
    mu_err = np.array([p["mu_err"] for p in fit["peaks"]])
    k = np.linspace(0, pe.max() + 0.5, 50)
    ax2.errorbar(pe, mu, yerr=mu_err, fmt="o", markersize=6, color=DATA_COLOR, label="peak means")
    ax2.plot(k, fit["gain"] * k + fit["pedestal"], color=FIT_COLOR, linewidth=2,
             label=f"this fit: {fit['gain']:.0f} ± {fit['gain_err']:.0f} per PE")
    if cal:
        ax2.plot(k, cal["gain"] * k + cal.get("pedestal", 0.0), color=REF_COLOR, linestyle="--",
                 linewidth=1.5, label=f"spe_calibration.yaml: {cal['gain']:.0f} per PE")
    ax2.set_xlabel("photoelectron number")
    ax2.set_ylabel("peak position [charge units]")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax2)

    fig.suptitle(f"{title}: single-photoelectron calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_npe_check(n_pe, path, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    v = n_pe[np.isfinite(n_pe)]
    ax.hist(v[(v > -1) & (v < 10)], bins=np.arange(-1, 10.0001, 0.05), histtype="step", color=DATA_COLOR)
    for k in range(0, 10):
        ax.axvline(k, color=REF_COLOR, linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("n_pe (calibrated); dotted lines at whole numbers")
    ax.set_ylabel("events per 0.05 PE")
    ax.set_title(f"{title}: peaks should sit on the dotted lines")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spectrum(n_pe, path, title, x_max):
    v = n_pe[np.isfinite(n_pe)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, x_max + 1, 1.0)
    for ax, log_y in ((ax1, True), (ax2, False)):
        ax.hist(v[(v >= 0) & (v < x_max)], bins=bins, histtype="step", color=DATA_COLOR)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("n_pe (photoelectrons)")
        ax.set_ylabel("events per PE")
        style(ax)
    ax2.set_xlim(5, x_max)
    # linear-scale panel: scale y to the region above the single-PE pileup
    above = v[(v > 5) & (v < x_max)]
    if len(above):
        h, _ = np.histogram(above, bins=np.arange(5, x_max + 1, 1.0))
        ax2.set_ylim(0, 1.15 * h.max())
    ax1.set_title("log scale")
    ax2.set_title("linear scale, above 5 PE")
    frac_over = float(np.mean(v >= x_max))
    fig.suptitle(f"{title}: energy spectrum ({len(v):,} events; {100 * frac_over:.2f}% above {x_max:g} PE)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return frac_over


def plot_timing(wf, amplitude, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    t, npe = wf["arrival_ns"], wf["arrival_npe"]
    med = float(np.median(t))
    lo, hi = np.percentile(t, [0.5, 99.5])
    pad = max(20.0, 0.5 * (hi - lo))
    rng = (max(0.0, lo - pad), hi + pad)

    axes[0].hist(t, bins=np.arange(rng[0], rng[1] + SAMPLE_NS, SAMPLE_NS), histtype="step", color=DATA_COLOR)
    axes[0].axvline(med, color=REF_COLOR, linestyle="--", linewidth=1, label=f"median {med:.0f} ns")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("arrival time: 50% of max [ns from charge-window start]")
    axes[0].set_ylabel("events")
    axes[0].legend(frameon=False)
    style(axes[0])

    h = axes[1].hist2d(npe, t, bins=[np.logspace(np.log10(0.5), np.log10(500), 80),
                                     np.linspace(rng[0], rng[1], 80)],
                       norm=LogNorm(), cmap="viridis")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_pe")
    axes[1].set_ylabel("arrival time [ns]")
    fig.colorbar(h[3], ax=axes[1], label="events")

    a = amplitude[np.isfinite(amplitude)]
    a_hi = np.percentile(a, 99.9)
    axes[2].hist(a[a < a_hi * 1.2], bins=300, histtype="step", color=DATA_COLOR)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("wf_amplitude (pulse height) [ADC]")
    axes[2].set_ylabel("events")
    style(axes[2])

    fig.suptitle(f"{title}: pulse timing ({len(t):,} sampled events) and pulse height (all events)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"arrival_median_ns": med,
            "arrival_p05_p95_ns": [float(x) for x in np.percentile(t, [5, 95])]}


def plot_spe(wf, path, title):
    spe = wf["spe_avg"]
    t_ns = np.arange(len(spe)) * SAMPLE_NS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(t_ns, spe, color=DATA_COLOR, linewidth=1.5)
    peak = int(np.argmax(spe))
    ax1.set_xlim(max(0, t_ns[peak] - 100), t_ns[peak] + 400)
    ax1.set_title("around the peak")
    ax2.plot(t_ns, np.clip(spe, 1e-6 * spe.max(), None), color=DATA_COLOR, linewidth=1)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4 * spe.max(), 2 * spe.max())
    ax2.set_title("full window, log scale")
    for ax in (ax1, ax2):
        ax.set_xlabel("time [ns from charge-window start]")
        ax.set_ylabel("average amplitude [ADC]")
        style(ax)
    fig.suptitle(f"{title}: average 1 PE pulse ({wf['n_spe']:,} events with 0.85 < n_pe < 1.15)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rate(t_s, path, title, bin_s):
    duration = float(t_s[-1])
    n_bins = max(int(np.ceil(duration / bin_s)), 1)
    counts, edges = np.histogram(t_s, bins=n_bins, range=(0, n_bins * bin_s))
    rate = counts / bin_s
    centers = 0.5 * (edges[:-1] + edges[1:])
    # the last bin is usually partly empty, so leave it out of the plot and the mean
    full = slice(0, -1) if n_bins > 1 else slice(None)
    mean_rate = float(rate[full].mean())

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(centers[full], rate[full], where="mid", color=DATA_COLOR, linewidth=1.5)
    ax.axhline(mean_rate, color=REF_COLOR, linestyle="--", linewidth=1, label=f"mean {mean_rate:.0f} /s")
    ax.set_xlabel("time since first recorded event [s]")
    ax.set_ylabel(f"recorded events per second ({bin_s:g} s bins)")
    ax.set_ylim(0, 1.2 * rate[full].max())
    ax.legend(frameon=False)
    style(ax)
    ax.set_title(f"{title}: recorded event rate ({len(t_s):,} events over {duration:.0f} s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"duration_s": duration, "mean_recorded_rate_hz": mean_rate,
            "rate_rms_over_mean": float(rate[full].std() / mean_rate) if mean_rate else None}


# ---------------------------------------------------------------------------

def main(args):
    run_dir = Path(args.run_dir)
    hit_files_all = sorted((run_dir / "hit").glob("*.parquet"))
    if not hit_files_all:
        raise FileNotFoundError(f"no hit files in {run_dir / 'hit'}")
    run_base = parse_run_info(hit_files_all[0].stem)[0]
    phase = load_phase(args.phases, run_base)
    label = args.label or run_base
    title = f"{label} ({phase})" if phase else label
    out = Path(args.output_dir) / label
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"run_base={run_base!r}, phase={phase}, output -> {out}")

    d, hit_files = load_hit(run_dir / "hit", args.n_files)
    summary = {"run_base": run_base, "phase": phase, "n_files": len(hit_files),
               "n_events": int(len(d["charge_total"]))}

    # 1. baseline
    summary["baseline"] = plot_baseline(d, out / f"{label}_baseline.png", title)

    # 2. charge fit
    cal = load_calibration(args.calibration, run_base)
    summary["calibration_yaml"] = cal
    if args.peak:
        windows = [parse_peak_arg(p) for p in args.peak]
    elif cal:
        windows = windows_from_gain(cal["gain"], cal.get("pedestal", 0.0), args.n_peaks)
    else:
        windows = None
        log.warning(f"no calibration for {run_base!r} in {args.calibration} and no --peak given: "
                    f"skipping the charge-fit plot. Run calibration/spe_search.py and calibration/spe_fit.py first.")
    if windows:
        gain_guess = cal["gain"] if cal else (windows[1][1] - windows[0][1])
        fit = fit_peaks(d["charge_total"], windows, bin_width=gain_guess / 50)
        if fit:
            plot_charge_fit(d["charge_total"], fit, cal, out / f"{label}_charge_fit.png", title,
                            n_peaks_shown=max(w[0] for w in windows))
            summary["refit"] = fit
            if cal:
                diff = 100 * (fit["gain"] - cal["gain"]) / cal["gain"]
                summary["refit_vs_yaml_gain_percent"] = diff
                log.info(f"gain: refit {fit['gain']:.0f} ± {fit['gain_err']:.0f}, "
                         f"YAML {cal['gain']:.0f} ({diff:+.2f}%)")
        else:
            log.warning("peak fit failed; charge-fit plot skipped")

    # 3-4. n_pe plots
    if "n_pe" in d:
        plot_npe_check(d["n_pe"], out / f"{label}_npe_check.png", title)
        summary["frac_above_spectrum_max"] = plot_spectrum(
            d["n_pe"], out / f"{label}_energy_spectrum.png", title, args.spectrum_max)
    else:
        log.warning("hit files have no n_pe (run not calibrated when the hit tier was made): "
                    "skipping n_pe check and energy spectrum")

    # 5-6. waveform sample
    if (run_dir / "dsp").exists() and "n_pe" in d:
        wf = sample_waveforms(run_dir / "dsp", hit_files, args.wf_files, args.wf_events)
        if wf is not None:
            summary["timing"] = plot_timing(wf, d.get("wf_amplitude", np.array([np.nan])),
                                            out / f"{label}_pulse_timing.png", title)
            plot_spe(wf, out / f"{label}_spe_pulse.png", title)
            summary["n_spe_waveforms"] = wf["n_spe"]
    else:
        log.warning("no dsp directory or no n_pe: skipping pulse timing and 1 PE pulse")

    # 7. event rate
    if not args.skip_raw and (run_dir / "raw").exists():
        t_s = load_timestamps(run_dir / "raw")
        if t_s is not None and len(t_s) > 1:
            summary["rate"] = plot_rate(t_s, out / f"{label}_event_rate.png", title, args.rate_bin_s)
            log.info(f"run length from timestamps: {summary['rate']['duration_s']:.0f} s "
                     f"(compare with time.real in compare_run_settings.py output)")
    else:
        log.info("raw tier not read: event-rate plot skipped")

    with open(out / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"done: plots and summary in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Run output directory containing raw/, dsp/, hit/")
    ap.add_argument("--label", default=None, help="Name for plots and output folder (default: run_base)")
    ap.add_argument("--output-dir", default="results/calibration")
    ap.add_argument("--calibration", default=str(REPO_ROOT / "config/spe_calibration.yaml"))
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--n-files", type=int, default=None, help="Use only the first N hit files (default: all)")
    ap.add_argument("--n-peaks", type=int, default=4, help="PE peaks to fit when windows come from the YAML gain")
    ap.add_argument("--peak", action="append", help="Explicit window PE:LO:HI (repeatable), as in calibration/spe_fit.py")
    ap.add_argument("--spectrum-max", type=float, default=300, help="Upper n_pe edge of the energy spectrum")
    ap.add_argument("--wf-files", type=int, default=3, help="dsp files to sample waveforms from")
    ap.add_argument("--wf-events", type=int, default=20000, help="events to sample per dsp file")
    ap.add_argument("--rate-bin-s", type=float, default=10.0, help="Time bin for the event-rate plot [s]")
    ap.add_argument("--skip-raw", action="store_true", help="Do not read the raw tier (no event-rate plot)")
    main(ap.parse_args())
