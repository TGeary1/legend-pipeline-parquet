"""
PEN model plus a rise time in the light.

Every event starts at the same place in its record (10% start spread
~2.5 ns in liquid), so the slow "toe" on the rising edge of the average pulse
is not a blur between events: the light itself ramps up over ~10-30 ns
instead of switching on at once. PEN alone cannot do that (an exponential
delay starts at full strength). A second exponential step in the chain can:
two exponential steps in a row start from zero and rise smoothly.

Model:  (singlet + triplet) (x) PEN exp(tau_pen) (x) rise exp(tau_rise) (x) R + floor
Free: t0, singlet and triplet amounts, tau_t, tau_pen, tau_rise, floor.
The rise step is causal (no light before t0), unlike a Gaussian smear.

Uses what lifetime_vs_time.py saved, like pen_model_fit.py.

Usage (from the repo root):
    python analysis/rise_model_fit.py --lifetime-dir results/lifetime_m180 \\
        --output-dir results/advisor/17_rise_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pen_model_fit import load, onset, style, SAMPLE_NS  # noqa: E402


def make_rise_model(R, n):
    s = np.arange(n, dtype=float)

    def model(_, t0, a_s, a_t, tau_t, tau_pen, tau_rise, floor):
        x = s - t0
        on = x >= 0
        xx = np.where(on, x, 0.0)
        pen = np.exp(-xx / tau_pen) / tau_pen
        if abs(tau_t - tau_pen) < 1e-6:
            trip = xx * np.exp(-xx / tau_t) / tau_t ** 2
        else:
            trip = (np.exp(-xx / tau_t) - np.exp(-xx / tau_pen)) / (tau_t - tau_pen)
        light = np.where(on, a_s * pen + a_t * trip + floor, 0.0)
        k = np.exp(-s / tau_rise)                       # causal rise step, full length
        light = fftconvolve(light, k / k.sum())[:n]
        return fftconvolve(light, R)[:n]
    return model


def fit_block(M, M_err, R, tau_t_start_ns=None, sys_frac=0.01):
    n = len(M)
    model = make_rise_model(R, n)
    err = np.sqrt(M_err ** 2 + (sys_frac * np.abs(M)) ** 2) + 1e-3
    total = M.sum()
    t0_guess = float(onset(M) - onset(R))
    tt_starts = [tau_t_start_ns / SAMPLE_NS] if tau_t_start_ns else [500, 1500]
    lo = [t0_guess - 30, 0, 0, 100, 2, 0.5, -np.inf]
    hi = [t0_guess + 30, np.inf, np.inf, 5000, 150, 50, np.inf]
    best = None
    for dt in (-8, 0, 8):
        for tt in tt_starts:
            for tp, tr in ((25, 3), (10, 8), (25, 10)):
                p0 = [t0_guess + dt, 0.3 * total, 0.6 * total, tt, tp, tr, 0]
                try:
                    p, _ = curve_fit(model, np.arange(n), M, p0=p0, sigma=err,
                                     absolute_sigma=True, bounds=(lo, hi), maxfev=20000)
                except (RuntimeError, ValueError):
                    continue
                curve = model(None, *p)
                chi2 = float(np.sum(((M - curve) / err) ** 2) / (n - len(p)))
                if best is None or chi2 < best["chi2_red"]:
                    best = {"p": p, "curve": curve, "chi2_red": chi2}
    if best is None:
        return None
    p = best["p"]
    return {"tau_t_ns": p[3] * SAMPLE_NS, "tau_pen_ns": p[4] * SAMPLE_NS, "tau_rise_ns": p[5] * SAMPLE_NS,
            "triplet_frac": p[2] / (p[1] + p[2]), "chi2_red": best["chi2_red"], "curve": best["curve"]}


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phases = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    blocks = [b for b in load(args.lifetime_dir, phases)
              if not args.only or b["label"] in args.only]

    print(f"\n{'block':<22}{'phase':<8}{'old chi2':>9}{'tau_t':>8}{'tau_pen':>9}{'rise':>8}"
          f"{'trip frac':>10}{'chi2':>8}")
    rows = []
    for b in blocks:
        old = b["old"] or {}
        r = fit_block(b["M"], b["M_err"], b["R"], tau_t_start_ns=old.get("tau_slow_ns"))
        rows.append((b, r))
        if r is None:
            print(f"{b['label']:<22}{b['phase']:<8}  fit failed")
            continue
        print(f"{b['label']:<22}{b['phase']:<8}{old.get('chi2_red', float('nan')):>9.2f}"
              f"{r['tau_t_ns']:>8.0f}{r['tau_pen_ns']:>9.1f}{r['tau_rise_ns']:>8.1f}"
              f"{r['triplet_frac']:>10.3f}{r['chi2_red']:>8.2f}")

    picks = [x for x in rows if x[0]["label"] in args.examples and x[1]] or [x for x in rows if x[1]][:3]
    fig, axes = plt.subplots(2, len(picks), figsize=(6 * len(picks), 7), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for j, (b, r) in enumerate(picks):
        t = np.arange(len(b["M"])) * SAMPLE_NS / 1000
        pk = b["M"].max()
        ax = axes[0][j]
        ax.plot(t, b["M"] / pk, color="#555555", linewidth=1.2, label="data")
        ax.plot(t, r["curve"] / pk, "--", color="#2a78d6", linewidth=1.4,
                label=f"PEN {r['tau_pen_ns']:.0f} ns + rise {r['tau_rise_ns']:.0f} ns, χ² {r['chi2_red']:.1f}")
        ax.set_xlim(0, 0.8)
        ax.set_title(f"{b['label']} ({b['phase']})")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
        ax = axes[1][j]
        ax.plot(t, (b["M"] - r["curve"]) / pk, color="#2a78d6", linewidth=1)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xlabel("time (μs)")
        style(ax)
    axes[0][0].set_ylabel("peak = 1")
    axes[1][0].set_ylabel("data − model")
    fig.tight_layout()
    fig.savefig(out / "rise_model_examples.png", dpi=120)
    plt.close(fig)
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime_m180")
    ap.add_argument("--output-dir", default="results/rise_model")
    ap.add_argument("--phases", default="config/run_phases.yaml")
    ap.add_argument("--only", nargs="*", help="Fit only these blocks (default: all)")
    ap.add_argument("--examples", nargs="*",
                    default=["260520_1340:0-9", "260520_1800:70-79", "260520_1239:20-29"])
    main(ap.parse_args())
