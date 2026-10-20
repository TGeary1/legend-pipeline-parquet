"""
Scintillation pulse shape by FFT deconvolution: liquid vs solid (and gas).

The measured average alpha pulse M(t) is the light's arrival-time profile
L(t) convolved with the PMT's single-photoelectron response R(t):

    M = L * R          ->  in frequency space:  M(f) = L(f) R(f)

so L can be recovered by dividing in frequency space and transforming back.
Plain division blows up noise wherever R(f) is small, so a Wiener filter
is used:  L(f) = M(f) R*(f) / (|R(f)|^2 + lambda).

This is the model-free cross-check of the forward fit: no exponentials are
assumed. Everything comes from what lifetime_vs_time.py already saved
(sums.pkl: average pulses and each run's 1 PE response), so it runs in
seconds on a login node.

Outputs (in --output-dir):
  pulse_shapes_measured.png     average pulses as recorded (peak = 1)
  pulse_shapes_deconvolved.png  recovered light vs time (log) and the
                                cumulative fraction of light vs time
  deconvolution_check.png       L re-convolved with R against M, per block

Usage (from the repo root):
    python analysis/pulse_shape_deconvolution.py --lifetime-dir results/lifetime \\
        --output-dir results/advisor/07_pulse_shape
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spe_deconvolution_analysis import wiener_deconvolve  # noqa: E402

SAMPLE_NS = 2.0
# (block label, legend text, colour)
DEFAULT_BLOCKS = [
    ("260520_1340:0-9", "liquid, early", "#eb6834"),
    ("260520_1800:70-79", "solid, evening", "#1baf7a"),
    ("260521_1023:0-9", "solid, next morning", "#0f6b4b"),
    ("260520_1239:0-9", "gas, start", "#7fb2ea"),
    ("260520_1239:20-29", "gas, cleanest", "#2a78d6"),
]


BIN = 20   # samples (40 ns) per bin when showing or fitting the recovered tail


def rebin(x, k=BIN):
    """Average over blocks of k samples. Deconvolution amplifies sample-to-
    sample noise; averaging neighbouring samples removes it again without
    changing the shape on the 100 ns-and-longer scales that matter here."""
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(axis=1)


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_blocks(lifetime_dir, wanted):
    """Average pulse, 1 PE response and forward-fit tau for each wanted block."""
    with open(Path(lifetime_dir) / "lifetime_vs_time.json") as f:
        summary = json.load(f)
    with open(Path(lifetime_dir) / "sums.pkl", "rb") as f:
        saved = pickle.load(f)
    max_jump = float(summary["settings"]["max_jump"])
    by_label = {g["label"]: (g, tot) for g, tot in zip(summary["groups"], saved["totals"])}

    blocks = []
    for label, text, color in wanted:
        if label not in by_label:
            print(f"block {label} not found in {lifetime_dir}; skipped")
            continue
        g, tot = by_label[label]
        M = tot["sum"][max_jump] / tot["n"][max_jump]
        R = saved["responses"][g["run"]][0]
        fit_tau = g["fit"]["tau_slow_ns"] if g["fit"] else None
        blocks.append({"label": label, "text": text, "color": color, "M": M, "R": R,
                       "fit_tau_ns": fit_tau})
    return blocks


def deconvolve(b, reg):
    """Wiener-deconvolve the 1 PE response out of the average pulse.
    reg = lambda as a fraction of the mean |R(f)|^2. The recovered light is
    normalised to unit area and its onset (first sample above 10% of the
    maximum) is taken as t = 0."""
    R, M = b["R"], b["M"]
    G = np.fft.rfft(np.r_[R, np.zeros(len(R))])
    lam = reg * np.mean(np.abs(G) ** 2)
    L = wiener_deconvolve(R, M, noise_power_estimate=lam, pad_factor=2)
    L = L / L.sum()
    onset = int(np.argmax(L > 0.1 * L.max()))
    b["L"], b["onset"] = L, onset
    b["M_check"] = fftconvolve(L, R)[:len(M)] * M.sum()
    return b


def summarise(b, tail_us):
    """Fraction of the light within 100 ns, 500 ns and 1 us of the onset, and
    an exponential-plus-constant fit to the recovered light over tail_us.
    The constant soaks up the slow ~1% floor seen in the solid, which would
    otherwise flatten a pure exponential and lengthen tau."""
    L, t0 = b["L"], b["onset"]
    cum = np.cumsum(L[t0:])
    frac = {ns: float(cum[min(int(ns / SAMPLE_NS), len(cum) - 1)]) for ns in (100, 500, 1000)}
    Lb = rebin(L[t0:])
    tb = (np.arange(len(Lb)) + 0.5) * BIN * SAMPLE_NS         
    sel = (tb >= tail_us[0] * 1000) & (tb <= tail_us[1] * 1000)
    t_sel, y_sel = tb[sel], Lb[sel]
    t_start = t_sel[0]

    def model(t, a, tau, c):
        return a * np.exp(-(t - t_start) / tau) + c

    try:
        popt, _ = curve_fit(model, t_sel, y_sel, p0=[max(y_sel[0], 1e-9), 1000.0, 0.0],
                            bounds=([0, 50, -np.inf], [np.inf, 20000, np.inf]), maxfev=20000)
        a, tau, c = popt
        floor_share = c * (len(L) - t0)        # L sums to 1, so this is the floor's share of the light
    except (RuntimeError, ValueError):
        tau, floor_share = None, None
    return frac, tau, floor_share


def plot_measured(blocks, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for b in blocks:
        t = np.arange(len(b["M"])) * SAMPLE_NS / 1000
        ax.semilogy(t, np.clip(b["M"] / b["M"].max(), 1e-4, None), color=b["color"], linewidth=1.4, label=b["text"])
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("time from start of charge window (μs)")
    ax.set_ylabel("average alpha pulse (peak = 1)")
    ax.set_title("Measured pulses: light shape smeared by the 1 PE response")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_deconvolved(blocks, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    for b in blocks:
        L, t0 = b["L"], b["onset"]
        t = (np.arange(len(L)) - t0) * SAMPLE_NS / 1000
        Lb = rebin(L[t0:])
        tb = (np.arange(len(Lb)) + 0.5) * BIN * SAMPLE_NS / 1000
        ax1.semilogy(tb, np.clip(Lb / Lb.max(), 1e-5, None), color=b["color"], linewidth=1.6, label=b["text"])
        ax2.plot(t[t0:], np.cumsum(L[t0:]), color=b["color"], linewidth=1.6, label=b["text"])
    ax1.set_xlim(0, 7)
    ax1.set_ylim(1e-3, 1.5)
    ax1.set_xlabel("time from light onset (μs)")
    ax1.set_ylabel("recovered light per 40 ns (first bin = 1)")
    ax1.set_title("Light arrival after removing the 1 PE response")
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 1.02)
    for x in (0.5, 1.0):
        ax2.axvline(x, color="#999999", linestyle=":", linewidth=1)
    ax2.set_xlabel("time from light onset (μs)")
    ax2.set_ylabel("fraction of the light emitted so far")
    ax2.set_title("Cumulative light (dotted: 0.5 and 1 μs)")
    for ax in (ax1, ax2):
        ax.legend(frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_check(blocks, path):
    fig, axes = plt.subplots(1, len(blocks), figsize=(5 * len(blocks), 4.2), squeeze=False)
    for ax, b in zip(axes[0], blocks):
        t = np.arange(len(b["M"])) * SAMPLE_NS / 1000
        peak = b["M"].max()
        ax.semilogy(t, np.clip(b["M"] / peak, 1e-4, None), color="#555555", linewidth=1.2, label="measured")
        ax.semilogy(t, np.clip(b["M_check"] / peak, 1e-4, None), "--", color=b["color"], linewidth=1.4,
                    label="recovered ⊗ 1 PE")
        ax.set_ylim(1e-3, 1.5)
        ax.set_title(b["text"])
        ax.set_xlabel("time (μs)")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0][0].set_ylabel("peak = 1")
    fig.suptitle("Check: the recovered light, convolved back with the 1 PE response, should reproduce the data")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = [deconvolve(b, args.reg) for b in load_blocks(args.lifetime_dir, DEFAULT_BLOCKS)]

    print(f"\n{'block':<22}{'light <100 ns':>14}{'<500 ns':>10}{'<1 us':>8}"
          f"{'tail tau (ns)':>15}{'floor':>8}{'fit tau (ns)':>14}{'check rms':>11}")
    for b in blocks:
        frac, tau, floor = summarise(b, args.tail_us)
        rms = np.sqrt(np.mean(((b["M_check"] - b["M"]) / b["M"].max()) ** 2))
        fit_tau = f"{b['fit_tau_ns']:.0f}" if b["fit_tau_ns"] else "-"
        tail = f"{tau:.0f}" if tau else "-"
        floor_txt = f"{floor:.3f}" if floor is not None else "-"
        print(f"{b['text']:<22}{frac[100]:>14.3f}{frac[500]:>10.3f}{frac[1000]:>8.3f}"
              f"{tail:>15}{floor_txt:>8}{fit_tau:>14}{rms:>11.4f}")

    plot_measured(blocks, out / "pulse_shapes_measured.png")
    plot_deconvolved(blocks, out / "pulse_shapes_deconvolved.png")
    plot_check(blocks, out / "deconvolution_check.png")
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime")
    ap.add_argument("--output-dir", default="results/pulse_shape")
    ap.add_argument("--reg", type=float, default=1e-4,
                    help="Wiener regularisation, as a fraction of mean |R(f)|^2 (larger = smoother, less sharp)")
    ap.add_argument("--tail-us", type=float, nargs=2, default=[1.0, 7.0],
                    help="Range after onset (us) for the single-exponential tail fit")
    main(ap.parse_args())
