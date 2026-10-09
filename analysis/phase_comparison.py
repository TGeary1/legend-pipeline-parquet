"""
Compare gas, liquid and solid argon: pulse shape discrimination (PSD) versus
brightness, and the Am-241 alpha peak used for the light yield.

Reads four hit-tier columns per event (n_pe, psd_param, lowest_adc,
max_jump_pe), applies two quality cuts, and fills histograms run by run, so
memory stays small even for ~60 million events.

Cuts:
  - saturation: lowest_adc must stay above the clipping floor + margin.
    Clipped pulses lose prompt charge, which biases both PSD and n_pe.
    The floor (~7,067 raw ADC) was measured on liquid; the script prints
    each run's lowest value so you can confirm all runs share it.
  - pileup: max_jump_pe (largest 20 ns rise after the main pulse, in units
    of one photon's pulse height) must be below --max-jump. Late argon
    light arrives one photon at a time, so a multi-PE jump is a second
    event landing in the record.

Alpha band and light yield:
  Am-241 emits 5.486 MeV alphas. In each phase the script finds the main
  high-brightness PSD population (20-150 PE), takes a band around it, fits
  a Gaussian to the n_pe spectrum of events in that band, and reports
      light yield = peak n_pe / 5.486 MeV   (detected PE per MeV, alpha)
  Check the band lines drawn on the 2D plot before trusting this: the band
  is found automatically, and if it locks onto the wrong population the
  light yield is meaningless. Override with --band PHASE:LO:HI.

Usage (from the repo root; light enough for a login node):
    python analysis/phase_comparison.py --output-dir results/phase_comparison

Outputs (in --output-dir):
  psd_vs_npe_by_phase.png     2D PSD vs n_pe, one panel per phase, band marked
  psd_bands_by_phase.png      PSD distributions in brightness slices, phases overlaid
  alpha_peak_by_phase.png     n_pe spectrum of the alpha band, with Gaussian fits
  psd_vs_npe_by_run.png       2D plot for each run separately, each with its own band
  alpha_peak_by_run.png       alpha spectrum for each run separately (consistency check)
  phase_comparison_summary.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AM241_ALPHA_MEV = 5.486   # main Am-241 alpha line (85%); 5.443 MeV (13%) is unresolved here
PHASES = ["gas", "liquid", "solid"]
# categorical colours in fixed order (gas, liquid, solid), plus a 4th for per-run plots
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#333333"

# histogram binning
LOG_NPE = np.logspace(np.log10(0.5), np.log10(500), 121)   # display
LIN_NPE = np.arange(0, 300.5, 0.5)                          # spectra and fits
PSD = np.linspace(-0.1, 1.1, 241)                           # 0.005 wide
NPE_SLICES = [(10, 30), (30, 100), (100, 300)]


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Filling histograms
# ---------------------------------------------------------------------------

def fill_run(run_dir, sat_floor, sat_margin, max_jump):
    """One pass over a run's hit files. Returns two 2D histograms of the
    events passing both cuts, plus cut bookkeeping."""
    files = sorted((Path(run_dir) / "hit").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no hit files in {run_dir}/hit")
    cols = ["n_pe", "psd_param", "lowest_adc", "max_jump_pe"]
    missing = [c for c in cols if c not in pq.ParquetFile(str(files[0])).schema_arrow.names]
    if missing:
        raise ValueError(f"{run_dir}: hit files lack {missing}; regenerate the hit tier first")

    h_log = np.zeros((len(LOG_NPE) - 1, len(PSD) - 1))
    h_lin = np.zeros((len(LIN_NPE) - 1, len(PSD) - 1))
    n_total = n_valid = n_sat = n_pile = 0
    lowest_seen = np.inf
    for f in files:
        t = pq.read_table(str(f), columns=cols)
        n_pe, psd, low, jump = (t.column(c).to_numpy() for c in cols)
        n_total += len(n_pe)
        valid = np.isfinite(psd) & np.isfinite(n_pe)
        sat = low < sat_floor + sat_margin
        pile = ~(jump < max_jump)              # NaN jump counts as failing
        lowest_seen = min(lowest_seen, float(np.nanmin(low)))
        n_valid += int(valid.sum())
        n_sat += int((valid & sat).sum())
        n_pile += int((valid & ~sat & pile).sum())
        keep = valid & ~sat & ~pile
        h_log += np.histogram2d(n_pe[keep], psd[keep], bins=[LOG_NPE, PSD])[0]
        h_lin += np.histogram2d(n_pe[keep], psd[keep], bins=[LIN_NPE, PSD])[0]
    return {
        "n_files": len(files), "n_total": n_total, "n_valid": n_valid,
        "n_saturated": n_sat, "n_pileup": n_pile, "n_kept": int(h_lin.sum()),
        "lowest_adc_min": lowest_seen, "h_log": h_log, "h_lin": h_lin,
    }


# ---------------------------------------------------------------------------
# Alpha band and peak fit
# ---------------------------------------------------------------------------

def find_band(h_lin, half_width, npe_range=(20, 150)):
    """PSD of the most populated band among bright events: the peak of the
    PSD distribution summed over npe_range, lightly smoothed."""
    i0, i1 = np.searchsorted(LIN_NPE, npe_range)
    prof = h_lin[i0:i1].sum(axis=0)
    smooth = np.convolve(prof, np.ones(5) / 5, mode="same")
    centers = 0.5 * (PSD[:-1] + PSD[1:])
    c = float(centers[np.argmax(smooth)])
    return c - half_width, c + half_width


def band_spectrum(h_lin, band):
    j0, j1 = np.searchsorted(PSD, band)
    return h_lin[:, j0:j1].sum(axis=1)


def gaussian(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def fit_peak(counts, min_npe=15):
    """Gaussian fit to the alpha hump, iterated over mu +/- 1.5 sigma so the
    single-photon spike at low n_pe and the high tail don't pull it.
    Returns None if the fit fails."""
    x = 0.5 * (LIN_NPE[:-1] + LIN_NPE[1:])
    above = x > min_npe
    if counts[above].sum() < 100:
        return None
    smooth = np.convolve(counts, np.ones(9) / 9, mode="same")
    mu = float(x[above][np.argmax(smooth[above])])
    s = 0.4 * mu
    p = None
    for _ in range(4):
        sel = (x > max(min_npe, mu - 1.5 * s)) & (x < mu + 1.5 * s) & (counts > 0)
        if sel.sum() < 5:
            return None
        try:
            p, cov = curve_fit(gaussian, x[sel], counts[sel], p0=[counts[sel].max(), mu, s],
                               sigma=np.sqrt(counts[sel]), absolute_sigma=True, maxfev=10000)
        except RuntimeError:
            return None
        mu, s = float(p[1]), abs(float(p[2]))
    return {"mu": mu, "mu_err": float(np.sqrt(cov[1, 1])), "sigma": s,
            "amp": float(p[0]), "fit_lo": float(max(min_npe, mu - 1.5 * s)), "fit_hi": float(mu + 1.5 * s),
            "light_yield_pe_per_mev": mu / AM241_ALPHA_MEV,
            "light_yield_err": float(np.sqrt(cov[1, 1])) / AM241_ALPHA_MEV,
            "poisson_sigma": float(np.sqrt(mu))}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_2d(data, bands, path):
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), sharey=True, squeeze=False)
    for ax, (phase, d) in zip(axes[0], data.items()):
        h = d["h_log"].T
        m = ax.pcolormesh(LOG_NPE, PSD, np.where(h > 0, h, np.nan), norm=LogNorm(), cmap="viridis")
        ax.set_xscale("log")
        lo, hi = bands[phase]
        for y in (lo, hi):
            ax.hlines(y, 20, 150, colors="white", linestyles="--", linewidth=1.2)
        ax.set_title(f"{phase} ({d['n_kept']:,} events after cuts)")
        ax.set_xlabel("n_pe (photoelectrons)")
        fig.colorbar(m, ax=ax, label="events")
    axes[0][0].set_ylabel("psd_param (prompt / total)")
    fig.suptitle("PSD vs brightness after saturation and pileup cuts (dashed: alpha band used for light yield)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_2d_runs(per_run, run_bands, path):
    """One 2D panel per run, in time order, each with its own band marked.
    Shows whether runs labelled with the same phase really look alike."""
    runs = sorted(per_run)
    ncol = 3
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.6 * nrow), sharey=True, squeeze=False)
    for ax, rb in zip(axes.flat, runs):
        h = per_run[rb]["h_log"].T
        m = ax.pcolormesh(LOG_NPE, PSD, np.where(h > 0, h, np.nan), norm=LogNorm(), cmap="viridis")
        ax.set_xscale("log")
        for y in run_bands[rb]:
            ax.hlines(y, 20, 150, colors="white", linestyles="--", linewidth=1.2)
        ax.set_title(f"{rb} ({per_run[rb]['phase']})")
        ax.set_xlabel("n_pe")
        fig.colorbar(m, ax=ax, label="events")
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("psd_param")
    fig.suptitle("PSD vs brightness for each run separately, after cuts (dashed: that run's own alpha band)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_slices(data, path):
    fig, axes = plt.subplots(1, len(NPE_SLICES), figsize=(5.5 * len(NPE_SLICES), 4.2), sharey=False)
    centers = 0.5 * (PSD[:-1] + PSD[1:])
    medians = {}
    for ax, (lo, hi) in zip(axes, NPE_SLICES):
        i0, i1 = np.searchsorted(LIN_NPE, (lo, hi))
        for k, (phase, d) in enumerate(data.items()):
            prof = d["h_lin"][i0:i1].sum(axis=0)
            if prof.sum() == 0:
                continue
            cdf = np.cumsum(prof) / prof.sum()
            med = float(np.interp(0.5, cdf, centers))
            medians.setdefault(phase, {})[f"{lo}-{hi}"] = med
            ax.step(centers, prof / prof.sum(), where="mid", color=COLORS[PHASES.index(phase)],
                    linewidth=1.5, label=f"{phase} (median {med:.2f})")
        ax.set_title(f"{lo} ≤ n_pe < {hi}")
        ax.set_xlabel("psd_param")
        ax.set_xlim(-0.05, 1.0)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0].set_ylabel("fraction of events per 0.005")
    fig.suptitle("PSD distribution by phase, in brightness slices (each curve normalised to 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return medians


def plot_alpha(spectra, fits, path, title, color_of):
    x = 0.5 * (LIN_NPE[:-1] + LIN_NPE[1:])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, counts in spectra.items():
        total = counts[x > 10].sum()
        if total == 0:
            continue
        norm = counts / total
        f = fits.get(name)
        label = name if f is None else f"{name}: peak {f['mu']:.1f} ± {f['mu_err']:.1f} PE"
        ax.step(x, norm, where="mid", color=color_of(name), linewidth=1.3, label=label)
        if f is not None:
            xx = np.linspace(f["fit_lo"], f["fit_hi"], 200)
            ax.plot(xx, gaussian(xx, f["amp"], f["mu"], f["sigma"]) / total,
                    color=color_of(name), linewidth=2.5, alpha=0.6)
    ax.set_xlim(0, 200)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_xlabel("n_pe (photoelectrons)")
    ax.set_ylabel("fraction of events above 10 PE, per 0.5 PE")
    ax.set_title(title)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        # key by digits only: an unquoted 260520_1340 is read by YAML as 2605201340
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}

    run_dirs = [Path(p) for p in args.run_dirs] if args.run_dirs else \
        sorted(p for p in Path(args.data_root).glob("data_2605*") if (p / "hit").exists())

    per_run = {}
    for rd in run_dirs:
        first = next((rd / "hit").glob("*.parquet"), None)
        if first is None:
            continue
        run_base = parse_run_info(first.stem)[0]
        phase = phase_db.get(run_base.replace("_", ""))
        if phase not in PHASES:
            log.warning(f"{rd.name}: run {run_base} has no phase in {args.phases}; skipped")
            continue
        log.info(f"filling {rd.name} ({phase})")
        r = fill_run(rd, args.sat_floor, args.sat_margin, args.max_jump)
        r.update(run_base=run_base, phase=phase)
        per_run[run_base] = r

    # combine runs into phases
    data = {}
    for phase in PHASES:
        runs = [r for r in per_run.values() if r["phase"] == phase]
        if not runs:
            continue
        data[phase] = {k: sum(r[k] for r in runs) for k in ("h_log", "h_lin", "n_kept")}

    # bookkeeping table
    print(f"\n{'run':<13}{'phase':<8}{'files':>6}{'events':>12}{'saturated':>11}{'pileup':>9}"
          f"{'kept':>8}{'min lowest_adc':>16}")
    for rb, r in sorted(per_run.items()):
        v = max(r["n_valid"], 1)
        print(f"{rb:<13}{r['phase']:<8}{r['n_files']:>6}{r['n_total']:>12,}"
              f"{100 * r['n_saturated'] / v:>10.2f}%{100 * r['n_pileup'] / v:>8.2f}%"
              f"{100 * r['n_kept'] / v:>7.1f}%{r['lowest_adc_min']:>16.0f}")
        if r["lowest_adc_min"] < args.sat_floor - 100:
            log.warning(f"{rb}: pulses reach {r['lowest_adc_min']:.0f}, well below the assumed floor "
                        f"{args.sat_floor}; this run may clip at a different level")

    # alpha band per phase
    overrides = {}
    for b in args.band or []:
        ph, lo, hi = b.split(":")
        overrides[ph] = (float(lo), float(hi))
    bands = {ph: overrides.get(ph) or find_band(d["h_lin"], args.band_half_width) for ph, d in data.items()}

    plot_2d(data, bands, out / "psd_vs_npe_by_phase.png")
    medians = plot_slices(data, out / "psd_bands_by_phase.png")

    phase_spectra = {ph: band_spectrum(d["h_lin"], bands[ph]) for ph, d in data.items()}
    phase_fits = {ph: fit_peak(s, args.fit_min) for ph, s in phase_spectra.items()}
    plot_alpha(phase_spectra, {k: v for k, v in phase_fits.items() if v},
               out / "alpha_peak_by_phase.png",
               "Alpha band n_pe spectrum by phase (Am-241, 5.486 MeV)",
               lambda n: COLORS[PHASES.index(n)])

    # every run on its own, each with its own automatically found band
    run_bands = {rb: overrides.get(rb) or find_band(r["h_lin"], args.band_half_width)
                 for rb, r in per_run.items()}
    plot_2d_runs(per_run, run_bands, out / "psd_vs_npe_by_run.png")
    run_spectra = {rb: band_spectrum(r["h_lin"], run_bands[rb]) for rb, r in sorted(per_run.items())}
    run_fits = {rb: fit_peak(sp, args.fit_min) for rb, sp in run_spectra.items()}
    run_names = list(run_spectra)
    plot_alpha(run_spectra, {k: v for k, v in run_fits.items() if v},
               out / "alpha_peak_by_run.png",
               "Each run separately: n_pe spectrum of its own alpha band",
               lambda n: COLORS[run_names.index(n) % len(COLORS)] if run_names.index(n) < len(COLORS)
               else ["#e87ba4", "#008300"][run_names.index(n) - len(COLORS)])

    print("\nAlpha band and light yield (detected PE per MeV of alpha energy, statistical errors only)")
    print(f"{'':<14}{'PSD band':>16}{'peak n_pe':>16}{'width σ':>10}{'√peak':>8}{'LY (PE/MeV)':>16}")
    for name, f in list(phase_fits.items()) + list(run_fits.items()):
        band = run_bands[name] if name in run_fits else bands[name]
        if f is None:
            print(f"{name:<14}{band[0]:>7.3f}-{band[1]:<8.3f}   fit failed")
            continue
        print(f"{name:<14}{band[0]:>7.3f}-{band[1]:<8.3f}{f['mu']:>9.2f} ± {f['mu_err']:<4.2f}"
              f"{f['sigma']:>10.1f}{f['poisson_sigma']:>8.1f}"
              f"{f['light_yield_pe_per_mev']:>9.2f} ± {f['light_yield_err']:.2f}")

    summary = {
        "cuts": {"sat_floor": args.sat_floor, "sat_margin": args.sat_margin, "max_jump_pe": args.max_jump},
        "runs": {rb: {k: v for k, v in r.items() if not k.startswith("h_")} for rb, r in per_run.items()},
        "alpha_band": {ph: list(b) for ph, b in bands.items()},
        "psd_median_by_slice": medians,
        "alpha_fit_by_phase": phase_fits,
        "alpha_band_by_run": {rb: list(b) for rb, b in run_bands.items()},
        "alpha_fit_by_run": run_fits,
    }
    with open(out / "phase_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"done: plots and summary in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT),
                    help="Folder holding the data_2605* run directories (default: repo root)")
    ap.add_argument("--run-dirs", nargs="*", help="Explicit run directories instead of --data-root discovery")
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--output-dir", default="results/phase_comparison")
    ap.add_argument("--sat-floor", type=float, default=7067, help="Raw ADC value where pulses clip")
    ap.add_argument("--sat-margin", type=float, default=50, help="Flag pulses within this many ADC of the floor")
    ap.add_argument("--max-jump", type=float, default=3.0,
                    help="Pileup cut: keep events whose largest late 20 ns rise is below this many PE")
    ap.add_argument("--band-half-width", type=float, default=0.08,
                    help="Half-width in PSD of the automatically found alpha band")
    ap.add_argument("--band", action="append",
                    help="Override a band: NAME:LO:HI, NAME a phase or a run, e.g. liquid:0.38:0.54 or 260520_1447:0.38:0.54")
    ap.add_argument("--fit-min", type=float, default=15,
                    help="Ignore n_pe below this in the alpha fit (photon-counting curves cross the band there)")
    main(ap.parse_args())
