"""
Refit the average alpha pulses with a physically motivated light model:

    argon light   = singlet (prompt, a few ns) + triplet (tau_t)
    at the PMT    = argon light (x) PEN re-emission (exponential, tau_pen)
    measured      = that (x) the measured single-PE response R  + flat floor

so the "intermediate" component of the old fit is replaced by the PEN
wavelength shifter's delay, which delays singlet and triplet light alike.
(The single-PE response comes from single photoelectrons, so it contains
the PMT and electronics but not the PEN delay; the PEN step belongs in the
model.) The singlet (~5-7 ns) is shorter than the 2 ns-sampled response can
resolve here, so it is modelled as instantaneous.

Each block is fitted twice:
  free   tau_pen free in every block: if the PEN reading is right it should
         come out the same in gas, liquid and solid
  fixed  tau_pen fixed (default 50 ns): the advisor's value
and the chi2/ndf of both is compared with the old model's (from the JSON).

Uses what lifetime_vs_time.py saved (sums.pkl, lifetime_vs_time.json);
no data files are re-read. Same weighting as the old fit: each sample's
error = statistical error of the average (+) 1% of the signal.

Usage (from the repo root):
    python analysis/pen_model_fit.py --lifetime-dir results/lifetime \\
        --output-dir results/advisor/12_pen_model
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

SAMPLE_NS = 2.0
PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_pen_model(R, n):
    """Model function of samples: t0 (onset, samples), a_s (singlet area),
    a_t (triplet area), tau_t and tau_pen (samples), floor (per sample)."""
    s = np.arange(n, dtype=float)

    def model(_, t0, a_s, a_t, tau_t, tau_pen, floor):
        x = s - t0
        on = x >= 0
        xx = np.where(on, x, 0.0)
        # argon light (singlet as an instantaneous spike, triplet exponential)
        # convolved with the PEN exponential, done analytically:
        #   delta (x) exp(pen)       = exp(-x/tau_pen)/tau_pen
        #   exp(tau_t) (x) exp(pen)  = (e^{-x/tau_t} - e^{-x/tau_pen}) / (tau_t - tau_pen)
        pen = np.exp(-xx / tau_pen) / tau_pen
        if abs(tau_t - tau_pen) < 1e-6:
            trip = xx * np.exp(-xx / tau_t) / tau_t ** 2
        else:
            trip = (np.exp(-xx / tau_t) - np.exp(-xx / tau_pen)) / (tau_t - tau_pen)
        light = np.where(on, a_s * pen + a_t * trip, 0.0) + np.where(on, floor, 0.0)
        return fftconvolve(light, R)[:n]
    return model

def onset(y, frac=0.1):
    """First sample where y reaches frac of its maximum."""
    return int(np.argmax(y > frac * y.max()))


def fit_block(M, M_err, model, R, tau_pen_fixed=None, tau_t_start_ns=None, sys_frac=0.01):
    """Weighted fit; tau_pen free unless tau_pen_fixed (ns) is given.

    The 1 PE response R has its own pre-onset delay (it is cut from the same
    charge window as the data), so the light's start time t0 is the data's
    onset MINUS R's onset, not the data's onset itself. Several starting
    t0, tau_pen and tau_t values are tried and the best chi2 is kept."""
    n = len(M)
    sigma = np.sqrt(M_err ** 2 + (sys_frac * np.abs(M)) ** 2) + 1e-3
    total = M.sum()
    t0_guess = float(onset(M) - onset(R))
    tt_starts = [tau_t_start_ns / SAMPLE_NS] if tau_t_start_ns else [500, 1500]
    if tau_pen_fixed is None:
        f = lambda s, t0, a_s, a_t, tt, tp, fl: model(None, t0, a_s, a_t, tt, tp, fl)
        lo = [t0_guess - 30, 0, 0, 100, 2, -np.inf]
        hi = [t0_guess + 30, np.inf, np.inf, 5000, 150, np.inf]
        starts = [[t0_guess + dt, 0.3 * total, 0.6 * total, tt, tp, 0]
                  for dt in (-8, 0, 8) for tt in tt_starts for tp in (10, 30)]
    else:
        tp = tau_pen_fixed / SAMPLE_NS
        f = lambda s, t0, a_s, a_t, tt, fl: model(None, t0, a_s, a_t, tt, tp, fl)
        lo = [t0_guess - 30, 0, 0, 100, -np.inf]
        hi = [t0_guess + 30, np.inf, np.inf, 5000, np.inf]
        starts = [[t0_guess + dt, 0.3 * total, 0.6 * total, tt, 0]
                  for dt in (-8, 0, 8) for tt in tt_starts]
    best = None
    for p0 in starts:
        try:
            p, _ = curve_fit(f, np.arange(n), M, p0=p0, sigma=sigma, absolute_sigma=True,
                             bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        curve = f(None, *p)
        chi2 = float(np.sum(((M - curve) / sigma) ** 2) / (n - len(p)))
        if best is None or chi2 < best["chi2_red"]:
            best = {"p": p, "curve": curve, "chi2_red": chi2}
    if best is None:
        return None
    p = best["p"]
    a_s, a_t, tau_t = p[1], p[2], p[3]
    tau_pen = p[4] * SAMPLE_NS if tau_pen_fixed is None else tau_pen_fixed
    return {"t0": p[0], "tau_t_ns": tau_t * SAMPLE_NS, "tau_pen_ns": tau_pen,
            "triplet_frac": a_t / (a_s + a_t), "chi2_red": best["chi2_red"], "curve": best["curve"]}

def load(lifetime_dir, phases):
    with open(Path(lifetime_dir) / "lifetime_vs_time.json") as f:
        summary = json.load(f)
    with open(Path(lifetime_dir) / "sums.pkl", "rb") as f:
        saved = pickle.load(f)
    max_jump = float(summary["settings"]["max_jump"])
    blocks = []
    for g, tot in zip(summary["groups"], saved["totals"]):
        n = tot["n"].get(max_jump, 0)
        if n < 2:
            continue
        M = tot["sum"][max_jump] / n
        var = np.maximum(tot["sumsq"][max_jump] / n - M ** 2, 0) if "sumsq" in tot else M * 0
        blocks.append({"label": g["label"], "run": g["run"], "phase": phases.get(g["run"].replace("_", ""), "?"),
                       "M": M, "M_err": np.sqrt(var / n), "R": saved["responses"][g["run"]][0],
                       "old": g["fit"]})
    return blocks


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phases = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    blocks = load(args.lifetime_dir, phases)

    print(f"\n{'block':<22}{'phase':<8}{'old: tau_t':>11}{'chi2':>8}"
          f"{'free: tau_t':>12}{'tau_pen':>9}{'trip frac':>10}{'chi2':>8}"
          f"{'fixed: tau_t':>13}{'trip frac':>10}{'chi2':>8}")
    rows = []
    for b in blocks:
        model = make_pen_model(b["R"], len(b["M"]))
        old = b["old"] or {}
        tt0 = old.get("tau_slow_ns")
        free = fit_block(b["M"], b["M_err"], model, b["R"], tau_t_start_ns=tt0)
        fixed = fit_block(b["M"], b["M_err"], model, b["R"], tau_pen_fixed=args.tau_pen,
                          tau_t_start_ns=tt0)
        rows.append((b, free, fixed))
        fmt = lambda r, k, f: (f"{r[k]:{f}}" if r else "-")
        print(f"{b['label']:<22}{b['phase']:<8}{old.get('tau_slow_ns', float('nan')):>11.0f}"
              f"{old.get('chi2_red', float('nan')):>8.2f}"
              f"{fmt(free, 'tau_t_ns', '.0f'):>12}{fmt(free, 'tau_pen_ns', '.1f'):>9}"
              f"{fmt(free, 'triplet_frac', '.3f'):>10}{fmt(free, 'chi2_red', '.2f'):>8}"
              f"{fmt(fixed, 'tau_t_ns', '.0f'):>13}{fmt(fixed, 'triplet_frac', '.3f'):>10}"
              f"{fmt(fixed, 'chi2_red', '.2f'):>8}")

    # summary plot: tau_pen (free) and chi2 of the three models, block by block
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, (b, free, fixed) in enumerate(rows):
        c = PHASE_COLOR.get(b["phase"], "#777777")
        if free:
            ax1.plot(i, free["tau_pen_ns"], "o", color=c, markersize=5)
            ax2.plot(i, free["chi2_red"], "o", color=c, markersize=5)
        if fixed:
            ax2.plot(i, fixed["chi2_red"], "s", mfc="none", color=c, markersize=5)
        if b["old"]:
            ax2.plot(i, b["old"]["chi2_red"], "x", color="#555555", markersize=5)
    ax1.axhline(args.tau_pen, color="#222222", linestyle="--", linewidth=1, label=f"{args.tau_pen:g} ns")
    ax1.set_ylabel("PEN τ, fitted freely (ns)")
    ax1.legend(frameon=False)
    for ph, c in PHASE_COLOR.items():
        ax1.plot([], [], "o", color=c, label=ph)
    ax1.legend(frameon=False)
    ax2.plot([], [], "o", color="#777777", label="PEN model, τ free")
    ax2.plot([], [], "s", mfc="none", color="#777777", label=f"PEN model, τ = {args.tau_pen:g} ns")
    ax2.plot([], [], "x", color="#555555", label="old model (intermediate component)")
    ax2.set_yscale("log")
    ax2.set_ylabel("χ²/ndf")
    ax2.set_xlabel("block (in the order of lifetime_vs_time.json)")
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "pen_model_summary.png", dpi=120)
    plt.close(fig)

    # example fits on a log scale
    picks = [r for r in rows if r[0]["label"] in args.examples] or rows[:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 4.5), squeeze=False)
    for ax, (b, free, fixed) in zip(axes[0], picks):
        t = np.arange(len(b["M"])) * SAMPLE_NS / 1000
        pk = b["M"].max()
        ax.semilogy(t, np.clip(b["M"] / pk, 1e-4, None), color="#555555", linewidth=1.2, label="data")
        if free:
            ax.semilogy(t, np.clip(free["curve"] / pk, 1e-4, None), "--", color="#2a78d6", linewidth=1.4,
                        label=f"PEN free: τ_pen {free['tau_pen_ns']:.0f} ns, χ² {free['chi2_red']:.1f}")
        if fixed:
            ax.semilogy(t, np.clip(fixed["curve"] / pk, 1e-4, None), ":", color="#eb6834", linewidth=1.6,
                        label=f"PEN {args.tau_pen:g} ns: χ² {fixed['chi2_red']:.1f}")
        ax.set_ylim(1e-3, 1.5)
        ax.set_xlim(0, 3)
        ax.set_title(b["label"])
        ax.set_xlabel("time (μs)")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0][0].set_ylabel("peak = 1")
    fig.tight_layout()
    fig.savefig(out / "pen_model_examples.png", dpi=120)
    plt.close(fig)
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime")
    ap.add_argument("--output-dir", default="results/pen_model")
    ap.add_argument("--phases", default="config/run_phases.yaml")
    ap.add_argument("--tau-pen", type=float, default=50.0, help="Fixed PEN time constant for the second fit (ns)")
    ap.add_argument("--examples", nargs="*",
                    default=["260520_1340:0-9", "260520_1800:70-79", "260520_1239:20-29"],
                    help="Blocks to show as example fits")
    main(ap.parse_args())
