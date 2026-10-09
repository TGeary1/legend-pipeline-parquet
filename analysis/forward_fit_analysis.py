"""
Forward-fit analysis of the scintillation time profile.

Measures how argon's light is emitted over time (fast and slow components,
slow lifetime, slow fraction) from one run's dsp + hit tier output.

Method, in order:
  1. Response R: average the single-photoelectron (1 PE) events. This is
     what one photon looks like after the PMT and electronics. Its noisy
     late tail is replaced with its own exponential extrapolation.
  2. Measurement M: average bright events (n_pe above a threshold). Dim
     events can't show the time profile, because the digitizer triggers on
     the first photon, pinning it at the same position every time.
  3. Pileup rejection: drop events whose waveform jumps up by several PE
     within 20 ns after the main pulse. Late argon light arrives one photon
     at a time (1 PE steps); a sudden multi-PE jump is a second, unrelated
     event. This cuts on jump SHAPE, not on the amount of late light, so it
     does not sculpt the slow component.
  4. Forward fit: model the light profile as
         prompt spike + intermediate exponential + slow exponential
         + event-correlated floor (switches on with the pulse),
     convolve it with R, shift by a timing offset t0, and fit to M.
     Nothing is divided, so no noise amplification and no regularization
     choice (unlike deconvolution).
  5. Sensitivity checks: refit with different pileup thresholds, different
     systematic allowances, and with the rise excluded. The spread of the
     slow lifetime across these is its systematic uncertainty.

Reporting notes (learned on the liquid run 260520_1340):
  - Prompt and intermediate components are degenerate at this resolution
    (both far faster than the ~640 ns electronics). Report their sum as
    "fast"; do not claim a separate intermediate component.
  - The +/- from the fit is statistical only. The spread across the
    sensitivity checks (typically ~ +/-150 ns) is the honest uncertainty.
  - Use the same n_pe threshold when comparing runs, so both averages carry
    the same trigger bias.
  - Sample spacing is 2 ns (confirmed from settings.xml). daq2lh5's
    CoMPASS decoder stores dt = 16 as a hardcoded default; ignore it.

Liquid run reference result (5 PE pileup cut, 1% systematic):
  slow tau ~ 1458 ns (range 1303-1656 across checks), slow fraction 0.047.

Usage:
    python analysis/forward_fit_analysis.py \\
        --dsp-dir data/dsp --hit-dir data/hit \\
        --file-glob "*liquid_1.parquet" \\
        --label liquid_260520_1340 --output-dir results/forward_fit

Needs n_pe in the hit files (run SPE calibration for the run first).
On NERSC, run on a compute node (salloc), not a login node: dsp waveform
files are ~1-2 GB each in memory.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_NS = 2.0  # DT5730 native 500 MS/s; see PIPELINE_OVERVIEW_PARQUET.md §7a


def us_to_samples(t_us):
    return int(round(t_us * 1000 / SAMPLE_NS))


# ---------------------------------------------------------------------------
# 1-2. Loading: response (1 PE average) and bright events, one file at a time
# ---------------------------------------------------------------------------

def load_run(dsp_dir, hit_dir, file_glob, n_pe_min, pe_window=(0.85, 1.15), max_files=None):
    """Loop over a run's files, keeping only what the analysis needs:
    a running sum of 1 PE waveforms (for R) and the bright waveforms (for M).
    Memory stays bounded by one file at a time plus the bright events."""
    dsp_files = sorted(Path(dsp_dir).glob(file_glob))
    if max_files:
        dsp_files = dsp_files[:max_files]
    if not dsp_files:
        raise FileNotFoundError(f"no dsp files matching {file_glob!r} in {dsp_dir}")

    spe_sum, n_1pe, n_total, bright = None, 0, 0, []
    for f in dsp_files:
        hit_path = Path(hit_dir) / f.name
        hit = pq.read_table(str(hit_path))
        if "n_pe" not in hit.column_names:
            raise ValueError(f"{hit_path} has no n_pe; run SPE calibration for this run first")
        n_pe = hit.column("n_pe").to_numpy()

        dsp = pq.read_table(str(f), columns=["wf_charge_window"])
        if dsp.num_rows != len(n_pe):
            raise ValueError(f"row mismatch between {f.name} dsp ({dsp.num_rows}) and hit ({len(n_pe)})")
        wcw = np.stack(pc.struct_field(dsp.column("wf_charge_window"), "values")
                       .to_numpy(zero_copy_only=False)).astype(np.float32)

        is_1pe = (n_pe > pe_window[0]) & (n_pe < pe_window[1])
        s = wcw[is_1pe].sum(axis=0, dtype=np.float64)
        spe_sum = s if spe_sum is None else spe_sum + s
        n_1pe += int(is_1pe.sum())
        bright.append(wcw[n_pe > n_pe_min])
        n_total += len(n_pe)
        log.info(f"{f.name}: {is_1pe.sum()} 1PE, {(n_pe > n_pe_min).sum()} bright")
        del wcw, dsp

    return spe_sum / n_1pe, n_1pe, np.concatenate(bright), n_total


def clean_response(spe, lo_us=1.0, hi_us=3.0):
    """Unit-area response R. Past hi_us the 1 PE average is mostly noise,
    so replace it with the exponential fitted over [lo_us, hi_us]."""
    samples = np.arange(len(spe))
    i_lo, i_hi = us_to_samples(lo_us), us_to_samples(hi_us)
    fit = linregress(samples[i_lo:i_hi], np.log(np.clip(spe[i_lo:i_hi], 1e-9, None)))
    R = spe.astype(float).copy()
    R[i_hi:] = np.exp(fit.intercept + fit.slope * samples[i_hi:])
    return R / R.sum(), -1 / fit.slope * SAMPLE_NS


# ---------------------------------------------------------------------------
# 3. Pileup tag
# ---------------------------------------------------------------------------

def max_jump_pe(wfs, spe_amp, lag=10, start=300, chunk=2000):
    """Largest rise over any `lag` samples (20 ns) after sample `start`
    (0.6 us, past the main peak), in units of one photoelectron's height.
    Processed in chunks to limit memory."""
    out = np.empty(len(wfs))
    for i in range(0, len(wfs), chunk):
        w = wfs[i:i + chunk]
        out[i:i + chunk] = (w[:, start + lag:] - w[:, start:-lag]).max(axis=1) / spe_amp
    return out


# ---------------------------------------------------------------------------
# 4. Model and fit
# ---------------------------------------------------------------------------

PARAM_NAMES = ["t0", "a_prompt", "a_inter", "tau_inter", "a_slow", "tau_slow", "floor"]


def make_model(R, n):
    samples = np.arange(n)

    def model(_, t0, a_p, a_i, tau_i, a_s, tau_s, floor):
        light = ((a_i / tau_i) * np.exp(-samples / tau_i)
                 + (a_s / tau_s) * np.exp(-samples / tau_s)
                 + floor)                      # floor switches on with the pulse
        light[0] += a_p                        # prompt spike
        conv = fftconvolve(light, R)[:n]
        return np.interp(samples - t0, samples, conv, left=0.0)   # timing offset

    return model


def run_fit(M, M_err, model, sys_frac=0.01, t_min_us=0.0):
    n = len(M)
    samples = np.arange(n)
    t_us = samples * SAMPLE_NS / 1000
    sigma = np.sqrt(M_err ** 2 + (sys_frac * np.abs(M)) ** 2) + 1e-3
    use = t_us >= t_min_us

    total = M.sum()
    p0    = [0,   0.2 * total, 0.2 * total, 50,  0.6 * total, 750,  0]
    lower = [-30, 0,           0,           5,   0,           300,  -np.inf]
    upper = [30,  np.inf,      np.inf,      300, np.inf,      5000, np.inf]

    popt, pcov = curve_fit(lambda s, *p: model(None, *p)[use], samples[use], M[use],
                           p0=p0, sigma=sigma[use], absolute_sigma=True,
                           bounds=(lower, upper), maxfev=20000)
    t0, a_p, a_i, tau_i, a_s, tau_s, floor = popt
    curve = model(None, *popt)
    fast = a_p + a_i
    floor_area = floor * n
    at_bound = [name for name, v, lo, hi in zip(PARAM_NAMES, popt, lower, upper)
                if np.isclose(v, lo, rtol=1e-4, atol=1e-9) or np.isclose(v, hi, rtol=1e-4, atol=1e-9)]

    return {
        "tau_slow_ns": tau_s * SAMPLE_NS,
        "tau_slow_stat_err_ns": float(np.sqrt(pcov[5, 5]) * SAMPLE_NS),
        "tau_inter_ns": tau_i * SAMPLE_NS,
        "t0_ns": t0 * SAMPLE_NS,
        "slow_frac": a_s / (fast + a_s),
        "floor_share": floor_area / (fast + a_s + floor_area),
        "chi2_red": float(np.sum(((M - curve) / sigma)[use] ** 2) / (use.sum() - len(popt))),
        "params_at_bound": at_bound,
        "curve": curve,
        "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_tail_comparison(t_us, spe, M, path, label):
    plt.figure()
    plt.semilogy(t_us, np.clip(spe / spe.max(), 1e-4, None), label="1 PE (electronics only)")
    plt.semilogy(t_us, np.clip(M / M.max(), 1e-4, None), label="clean bright average")
    plt.xlabel("time from start of charge window (μs)")
    plt.ylabel("normalized amplitude")
    plt.ylim(1e-4, 2)
    plt.title(label)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_jump_hist(jumps, threshold, path, label):
    plt.figure()
    plt.hist(jumps, bins=200, range=(0, 30))
    plt.axvline(threshold, color="r", ls="--", label=f"cut: {threshold} PE")
    plt.yscale("log")
    plt.xlabel("largest 20 ns rise after main peak (PE)")
    plt.ylabel("events")
    plt.title(label)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_fit(t_us, M, fit, path, label):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), height_ratios=[3, 1])
    ax1.semilogy(t_us, np.clip(M / M.max(), 1e-4, None), label="clean bright average (data)")
    ax1.semilogy(t_us, np.clip(fit["curve"] / M.max(), 1e-4, None), "r--", label="model ⊗ response")
    ax1.set_ylabel("normalized amplitude")
    ax1.set_ylim(1e-4, 2)
    ax1.set_title(f"{label}: τ_slow = {fit['tau_slow_ns']:.0f} ns, slow frac = {fit['slow_frac']:.3f}")
    ax1.legend()
    ax2.plot(t_us, (M - fit["curve"]) / fit["sigma"], lw=0.5)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("residual (σ)")
    ax2.set_xlabel("time from start of charge window (μs)")
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label

    spe, n_1pe, bright, n_total = load_run(args.dsp_dir, args.hit_dir, args.file_glob,
                                           args.n_pe_min, max_files=args.max_files)
    n = bright.shape[1]
    t_us = np.arange(n) * SAMPLE_NS / 1000
    log.info(f"{n_total} events total; {n_1pe} 1PE; {len(bright)} with n_pe > {args.n_pe_min}")

    R, tau_elec_ns = clean_response(spe)
    model = make_model(R, n)
    log.info(f"electronics τ from 1PE response: {tau_elec_ns:.0f} ns")

    jumps = max_jump_pe(bright, spe.max())
    plot_jump_hist(jumps, args.jump_pe, out / f"{label}_jump_hist.png", label)

    def clean_average(thr):
        keep = jumps < thr
        return bright[keep].mean(axis=0), bright[keep].std(axis=0) / np.sqrt(keep.sum()), keep.mean()

    M, M_err, kept = clean_average(args.jump_pe)
    log.info(f"pileup cut < {args.jump_pe} PE keeps {kept:.1%} of bright events")
    plot_tail_comparison(t_us, spe, M, out / f"{label}_tail_comparison.png", label)

    base = run_fit(M, M_err, model, sys_frac=args.sys_frac)
    plot_fit(t_us, M, base, out / f"{label}_forward_fit.png", label)

    checks = []
    for thr in sorted({3, args.jump_pe, 10}):
        Mx, Mx_err, kx = clean_average(thr)
        r = run_fit(Mx, Mx_err, model, sys_frac=args.sys_frac)
        checks.append({"check": f"jump < {thr} PE (kept {kx:.1%})", **r})
    for sf in sorted({0.005, args.sys_frac, 0.02}):
        r = run_fit(M, M_err, model, sys_frac=sf)
        checks.append({"check": f"sys {sf:.1%}", **r})
    r = run_fit(M, M_err, model, sys_frac=args.sys_frac, t_min_us=0.3)
    checks.append({"check": "tail only (t > 0.3 us)", **r})

    taus = [c["tau_slow_ns"] for c in checks]

    print(f"\n=== {label} ===")
    print(f"events: {n_total} total, {n_1pe} 1PE, {len(bright)} bright (n_pe > {args.n_pe_min}), "
          f"{kept:.1%} pass pileup cut")
    print(f"electronics τ: {tau_elec_ns:.0f} ns")
    print(f"baseline: τ_slow = {base['tau_slow_ns']:.0f} ± {base['tau_slow_stat_err_ns']:.0f} (stat) ns, "
          f"slow frac = {base['slow_frac']:.3f}, floor share = {base['floor_share']:.3f}, "
          f"χ²/ndf = {base['chi2_red']:.2f}")
    if base["params_at_bound"]:
        print(f"  parameters at a bound: {base['params_at_bound']}")
    print("sensitivity checks:")
    for c in checks:
        print(f"  {c['check']:<28} τ_slow = {c['tau_slow_ns']:6.0f} ns, slow frac = {c['slow_frac']:.3f}")
    print(f"τ_slow systematic range: {min(taus):.0f}–{max(taus):.0f} ns")

    strip = lambda d: {k: v for k, v in d.items() if k not in ("curve", "sigma")}
    summary = {
        "label": label, "n_total": n_total, "n_1pe": n_1pe, "n_bright": len(bright),
        "n_pe_min": args.n_pe_min, "jump_pe": args.jump_pe, "pileup_kept": kept,
        "tau_elec_ns": tau_elec_ns, "baseline": strip(base),
        "checks": [strip(c) for c in checks],
        "tau_slow_range_ns": [min(taus), max(taus)],
    }
    with open(out / f"{label}_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"results written to {out / f'{label}_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsp-dir", required=True)
    p.add_argument("--hit-dir", required=True)
    p.add_argument("--file-glob", default="*.parquet", help="which files in the dirs belong to this run")
    p.add_argument("--label", required=True)
    p.add_argument("--output-dir", default="results/forward_fit")
    p.add_argument("--n-pe-min", type=float, default=30, help="bright-event threshold; keep equal across compared runs")
    p.add_argument("--jump-pe", type=float, default=5, help="pileup cut: max 20 ns rise, in PE")
    p.add_argument("--sys-frac", type=float, default=0.01, help="per-sample systematic, fraction of M")
    p.add_argument("--max-files", type=int, default=None, help="limit files for quick tests")
    main(p.parse_args())