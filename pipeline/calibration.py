# pipeline/calibration.py
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_spe_calibration(charge_total, peak_windows, bins=300, hist_range=None):
    """
    Derive SPE gain from the low-charge, self-calibrating population of a
    physics run's charge_total distribution.

    Parameters
    ----------
    charge_total : array of per-event integrated charge (from hit tier)
    peak_windows : list of (pe_number, lo, hi) tuples — one per resolvable
        peak, giving the PE number it corresponds to and a (lo, hi) charge
        range to restrict the fit to. Windows must be chosen by inspecting
        a histogram first (see notebook/script workflow) — this is
        inherently per-run/per-PMT-setting, not automatic.
    bins, hist_range : passed to np.histogram

    Returns
    -------
    dict with gain, gain_err, pedestal, pedestal_err, r_squared,
    and per-peak fit results
    """
    if hist_range is None:
        hist_range = (0, max(w[2] for w in peak_windows) * 1.2)

    counts, edges = np.histogram(charge_total[charge_total > 0], bins=bins, range=hist_range)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    peak_fits = {}
    for pe_number, lo, hi in peak_windows:
        mask = (bin_centers > lo) & (bin_centers < hi)
        if mask.sum() < 5:
            raise ValueError(f"Too few bins in window ({lo}, {hi}) for PE={pe_number} — check the range")
        p0 = [counts[mask].max(), (lo + hi) / 2, (hi - lo) / 6]
        popt, pcov = curve_fit(gaussian, bin_centers[mask], counts[mask], p0=p0)
        peak_fits[pe_number] = {
            "amplitude": popt[0], "mean": popt[1], "sigma": popt[2],
            "mean_err": np.sqrt(pcov[1, 1]),
        }

    pe_numbers = np.array(sorted(peak_fits.keys()))
    peak_positions = np.array([peak_fits[n]["mean"] for n in pe_numbers])

    fit = linregress(pe_numbers, peak_positions)

    return {
        "gain": fit.slope,
        "gain_err": fit.stderr,
        "pedestal": fit.intercept,
        "pedestal_err": fit.intercept_stderr,
        "r_squared": fit.rvalue ** 2,
        "n_peaks_used": len(pe_numbers),
        "peak_fits": peak_fits,
    }
