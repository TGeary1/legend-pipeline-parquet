"""
SPE deconvolution and decay-constant analysis.

Pipeline:
  1. Select clean single-photoelectron (1PE) events using the run's SPE
     calibration (config/spe_calibration.yaml) and average them into a
     "superpulse" — the PMT's own single-photoelectron response function.
  2. Average all events in the run to get the full-energy average waveform.
  3. Deconvolve the PMT response out of the full-energy average via FFT
     division (Wiener-regularized, edge-tapered, zero-padded to avoid
     circular-convolution wraparound), recovering the true underlying
     scintillation light pulse shape.
  4. Diagnose whether a single exponential is an adequate description of
     the decay (it is not, for the liquid run validated so far — see
     window-sensitivity check) and fit a biexponential model instead.

Validated on the liquid run (260520_1340_liquid_1): tau1 ~ 276 ns
(likely an instrumental/deconvolution-residual component, not read as a
fundamental scintillation lifetime), tau2 ~ 2355 ns (physically plausible
as an argon triplet-lifetime measurement, order-of-magnitude consistent
with liquid-argon literature ~1.5 microseconds, though notably longer —
worth cross-checking against the SAr run before treating as final).

IMPORTANT CAVEATS, carried over from the pipeline's documentation:
  - Waveform sample spacing is confirmed 16 ns/sample (CoMPASS onboard
    8x presumming), not the native 2 ns/sample digitizer rate. All time
    conversions in this script use SAMPLE_NS = 16.0. Do not assume 2 ns
    if reusing this code elsewhere.
  - Argon's ~6 ns singlet lifetime is NOT resolvable at 16 ns/sample,
    under any analysis choice. Do not interpret any fitted timescale from
    this script as "the singlet component."
  - The Wiener regularization's noise_power_estimate is a real tuning
    choice, not a fixed constant — see resolved-region-sensitivity notes
    inline. Results should be checked against at least one alternative
    noise estimate before being treated as final.

Usage:
    python spe_deconvolution_analysis.py \
  --dsp-file .../SAr_dsp_file.parquet \
  --hit-file .../SAr_hit_file.parquet \
  --output-dir results/sar_deconvolution \
  --label sar_1450pm
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.stats import linregress
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_NS = 16.0  # confirmed from WaveformTable.dt; CoMPASS onboard 8x presumming of the
                  # DT5730's native 2 ns/sample (500 MS/s) rate. See PIPELINE_OVERVIEW_PARQUET.md §7a.


def load_wf_charge_window(dsp_path):
    """Extract wf_charge_window as a plain (n_events, n_samples) numpy array."""
    t = pq.read_table(str(dsp_path))
    wcw_col = pc.struct_field(t.column("wf_charge_window"), "values")
    return np.stack(wcw_col.to_numpy(zero_copy_only=False)), t


def build_spe_superpulse(wcw, n_pe, pe_lo=0.85, pe_hi=1.15):
    """Average waveforms selected as clean single-photoelectron events.

    Uses fixed-position (trigger-relative) averaging rather than per-event
    alignment — validated by comparing the superpulse's rise sharpness
    against individual single-event rises (no visible trigger-walk blur
    found for this dataset). Revisit this choice if applying to a
    different detector/trigger configuration.
    """
    mask = (n_pe > pe_lo) & (n_pe < pe_hi)
    n_selected = mask.sum()
    if n_selected < 1000:
        log.warning(f"Only {n_selected} 1PE candidate events selected — superpulse may be noisy")
    else:
        log.info(f"{n_selected} candidate 1PE events selected for superpulse")
    return wcw[mask].mean(axis=0), n_selected


def edge_taper(signal, taper_len=200):
    """Force a smooth transition to zero over only the last `taper_len`
    samples, leaving the rest of the signal (including any real late-time
    decay) untouched. Using a full-signal window (e.g. plain np.hanning
    across the whole trace) is a real mistake here — it suppresses the
    genuine physics in the late-time region along with the discontinuity
    at the array edge. Only tapering the edge avoids this."""
    tapered = signal.copy().astype(float)
    ramp = np.hanning(2 * taper_len)[taper_len:]
    tapered[-taper_len:] *= ramp
    return tapered


def wiener_deconvolve(response, signal, noise_power_estimate=None, pad_factor=2):
    """Deconvolve `response` (e.g. the SPE superpulse) out of `signal`
    (e.g. the full-energy average waveform) via Wiener-regularized FFT
    division, with edge-tapering and zero-padding to avoid circular-
    convolution wraparound artifacts.

    If noise_power_estimate is None, derives one from a placeholder
    fraction of the response spectrum's power. Prefer passing a real
    estimate derived from measured baseline noise (bl_sig) when available
    — see main() for the recommended approach.
    """
    n = len(response)
    pad_len = pad_factor * n

    response_tapered = edge_taper(response)
    signal_tapered = edge_taper(signal)

    response_padded = np.zeros(pad_len)
    response_padded[:n] = response_tapered
    signal_padded = np.zeros(pad_len)
    signal_padded[:n] = signal_tapered

    G = np.fft.rfft(response_padded)
    H = np.fft.rfft(signal_padded)

    if noise_power_estimate is None:
        noise_power_estimate = 0.01 * np.mean(np.abs(G) ** 2)
        log.warning("No noise_power_estimate provided — using a placeholder fraction of "
                    "|G|^2. Prefer deriving this from measured bl_sig instead.")

    wiener_filter = np.conj(G) / (np.abs(G) ** 2 + noise_power_estimate)
    deconvolved_freq = H * wiener_filter
    deconvolved = np.fft.irfft(deconvolved_freq, n=pad_len)[:n]
    return deconvolved


def scan_single_exp_windows(deconvolved, peak_idx, windows):
    """Fit a single exponential over several candidate windows past the
    peak and report tau + R^2 for each. Large variation in tau across
    windows is evidence the decay is NOT well-described by a single
    exponential (see biexponential fit instead) rather than evidence of
    a bad fit region choice."""
    results = []
    for start_offset, end in windows:
        region = np.abs(deconvolved[peak_idx + start_offset: end])
        region = region[region > 0]
        if len(region) < 10:
            continue
        log_region = np.log(region)
        x = np.arange(len(log_region))
        fit = linregress(x, log_region)
        tau_ns = -1 / fit.slope * SAMPLE_NS
        results.append({
            "window": (peak_idx + start_offset, end),
            "tau_ns": tau_ns,
            "r_squared": fit.rvalue ** 2,
        })
    return results


def biexp(x, a1, tau1, a2, tau2):
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2)


def fit_biexponential(deconvolved, peak_idx, fit_end=300, p0=None):
    """Fit a two-component exponential decay to the deconvolved pulse.

    Returns a dict with each component's tau (in ns) and amplitude, plus
    the parameter covariance matrix for uncertainty estimation.
    """
    decay_region = np.abs(deconvolved[peak_idx + 5: fit_end])
    x = np.arange(len(decay_region))

    if p0 is None:
        p0 = [decay_region[0], 20, decay_region[0] * 0.1, 100]

    popt, pcov = curve_fit(biexp, x, decay_region, p0=p0, maxfev=10000)
    a1, tau1, a2, tau2 = popt
    perr = np.sqrt(np.diag(pcov))

    return {
        "tau1_ns": tau1 * SAMPLE_NS, "tau1_ns_err": perr[1] * SAMPLE_NS, "amp1": a1,
        "tau2_ns": tau2 * SAMPLE_NS, "tau2_ns_err": perr[3] * SAMPLE_NS, "amp2": a2,
        "popt": popt, "pcov": pcov, "x": x, "decay_region": decay_region,
    }


def run_analysis(dsp_path, hit_path, output_dir, label,
                  pe_lo=0.85, pe_hi=1.15, taper_len=200,
                  single_exp_windows=((15, 150), (15, 200), (20, 150), (20, 250), (30, 200)),
                  biexp_fit_end=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {dsp_path}")
    wcw, dsp_t = load_wf_charge_window(dsp_path)
    hit_t = pq.read_table(str(hit_path))
    assert dsp_t.num_rows == hit_t.num_rows, "dsp/hit row count mismatch — not from the same processing run"

    if "n_pe" not in hit_t.column_names:
        raise ValueError(f"{hit_path} has no n_pe column — run SPE calibration for this run first "
                          f"(config/spe_calibration.yaml) before attempting deconvolution")

    n_pe = hit_t.column("n_pe").to_numpy()
    bl_sig = dsp_t.column("bl_sig").to_numpy()

    # --- Step 1: SPE superpulse ---
    spe_superpulse, n_1pe = build_spe_superpulse(wcw, n_pe, pe_lo, pe_hi)

    plt.figure()
    plt.plot(spe_superpulse)
    plt.xlabel("sample (relative to wf_charge_window start)")
    plt.ylabel("ADC (baseline-subtracted)")
    plt.title(f"{label}: 1PE superpulse, n={n_1pe} events")
    plt.savefig(output_dir / f"{label}_spe_superpulse.png")
    plt.close()

    # --- Step 2: full-energy average ---
    avg_wf_all = wcw.mean(axis=0)

    # --- Step 3: deconvolution ---
    bl_sig_typical = bl_sig.mean()
    pad_len = 2 * len(spe_superpulse)
    noise_power_estimate = (bl_sig_typical ** 2) * pad_len
    log.info(f"Using bl_sig-derived noise_power_estimate: {noise_power_estimate:.3e} "
             f"(from mean bl_sig={bl_sig_typical:.3f})")

    deconvolved = wiener_deconvolve(spe_superpulse, avg_wf_all, noise_power_estimate=noise_power_estimate)

    peak_idx = int(np.argmax(np.abs(deconvolved)))
    log.info(f"Deconvolved pulse peaks at sample {peak_idx}")

    plt.figure()
    plt.plot(deconvolved)
    plt.xlabel("sample")
    plt.ylabel("deconvolved amplitude")
    plt.title(f"{label}: deconvolved pulse")
    plt.savefig(output_dir / f"{label}_deconvolved.png")
    plt.close()

    plt.figure()
    plt.semilogy(np.abs(deconvolved) + 1e-6)
    plt.xlabel("sample")
    plt.ylabel("|deconvolved amplitude| (log scale)")
    plt.title(f"{label}: deconvolved pulse, log scale")
    plt.savefig(output_dir / f"{label}_deconvolved_log.png")
    plt.close()

    # --- Step 4a: single-exponential window-sensitivity diagnostic ---
    single_exp_results = scan_single_exp_windows(deconvolved, peak_idx, single_exp_windows)
    log.info("Single-exponential window sensitivity (large tau variation => not a single exponential):")
    for r in single_exp_results:
        log.info(f"  window {r['window']}: tau={r['tau_ns']:.1f} ns, R^2={r['r_squared']:.4f}")

    taus = [r["tau_ns"] for r in single_exp_results]
    tau_spread = (max(taus) - min(taus)) / np.mean(taus) if taus else None
    if tau_spread is not None and tau_spread > 0.15:
        log.warning(f"Single-exponential tau varies by {tau_spread:.0%} across fit windows — "
                    f"decay is likely multi-component; treat single-exponential tau as unreliable")

    # --- Step 4b: biexponential fit ---
    biexp_result = fit_biexponential(deconvolved, peak_idx, fit_end=biexp_fit_end)
    log.info(f"Biexponential fit: tau1={biexp_result['tau1_ns']:.1f}±{biexp_result['tau1_ns_err']:.1f} ns "
              f"(amp={biexp_result['amp1']:.3f}), "
              f"tau2={biexp_result['tau2_ns']:.1f}±{biexp_result['tau2_ns_err']:.1f} ns "
              f"(amp={biexp_result['amp2']:.3f})")

    plt.figure()
    plt.semilogy(biexp_result["x"], biexp_result["decay_region"], 'o', markersize=2, label="data")
    plt.semilogy(biexp_result["x"], biexp(biexp_result["x"], *biexp_result["popt"]), 'r-', label="biexponential fit")
    plt.xlabel("sample (relative to peak)")
    plt.ylabel("deconvolved amplitude")
    plt.legend()
    plt.title(f"{label}: biexponential decay fit")
    plt.savefig(output_dir / f"{label}_biexp_fit.png")
    plt.close()

    return {
        "label": label,
        "n_1pe": n_1pe,
        "peak_idx": peak_idx,
        "single_exp_results": single_exp_results,
        "biexp_result": biexp_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsp-file", required=True, help="Path to a dsp-tier parquet file")
    parser.add_argument("--hit-file", required=True, help="Path to the matching hit-tier parquet file (must have n_pe)")
    parser.add_argument("--output-dir", default="results/deconvolution", help="Directory for output plots")
    parser.add_argument("--label", required=True, help="Short label for this run, used in filenames/titles")
    args = parser.parse_args()

    result = run_analysis(args.dsp_file, args.hit_file, args.output_dir, args.label)

    print(f"\n=== Summary for {result['label']} ===")
    print(f"1PE events used for superpulse: {result['n_1pe']}")
    print(f"Deconvolved pulse peak at sample: {result['peak_idx']}")
    br = result["biexp_result"]
    print(f"Biexponential fit:")
    print(f"  tau1 = {br['tau1_ns']:.1f} +/- {br['tau1_ns_err']:.1f} ns")
    print(f"  tau2 = {br['tau2_ns']:.1f} +/- {br['tau2_ns_err']:.1f} ns")
    print(f"\nCAUTION: tau1 is likely an instrumental/deconvolution-residual component, not a "
          f"fundamental scintillation lifetime. tau2 is the more physically interesting value, "
          f"but should be cross-checked against a different noise_power_estimate and against "
          f"the same analysis run on other datasets before being treated as final. See module "
          f"docstring and PIPELINE_OVERVIEW_PARQUET.md for caveats (16 ns/sample presumming, "
          f"singlet component not resolvable).")
