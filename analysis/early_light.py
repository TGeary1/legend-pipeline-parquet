"""
What does the light look like in its first few hundred ns? (model-free)

Every fitted light model so far leaves the same residual: the data have MORE
light in the first ~50 ns than the model, then a little less after the peak.
Adding a rise time made it worse, so the early light is faster than our
models allow, not slower. Before guessing another model, look at the light
directly: undo the 1 PE response R by Wiener deconvolution at full 2 ns
resolution, and see what shape the earliest light has.

  L(f) = M(f) R*(f) / (|R(f)|^2 + lam),   lam = reg x mean |R(f)|^2

Three regularisation strengths are shown: features that survive all three
are real; wiggles that change with reg are deconvolution noise.

For each block it reports, from the middle regularisation:
  - the fraction of the light (out to --total-us) arriving within 10, 20,
    50 and 100 ns of the light's start
  - an exponential fitted to the light from 6 ns after its peak to
    --early-ns, plus a constant for the slow light underneath: the decay
    time of the early light (PEN alone would give ~30-60 ns; an argon
    singlet seen directly would give ~5-7 ns, unresolvable here)

Uses what lifetime_vs_time.py saved; no data files are re-read.

Usage (from the repo root):
    python analysis/early_light.py --lifetime-dir results/lifetime_m180 \\
        --output-dir results/advisor/18_early_light
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pen_model_fit import load, style, SAMPLE_NS  # noqa: E402

PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}


def deconvolve(M, R, reg):
    """Wiener deconvolution of M by R (zero-padded to twice the length)."""
    n = len(M)
    nfft = 2 * n
    Mf, Rf = np.fft.rfft(M, nfft), np.fft.rfft(R, nfft)
    lam = reg * np.mean(np.abs(Rf) ** 2)
    return np.fft.irfft(Mf * np.conj(Rf) / (np.abs(Rf) ** 2 + lam), nfft)[:n]


def light_start(L):
    """Sample where the light first reaches 20% of its peak (interpolated)."""
    p = int(np.argmax(L))
    level = 0.2 * L[p]
    below = np.flatnonzero(L[:p] < level)
    j = below[-1] if len(below) else 0
    return j + (level - L[j]) / (L[j + 1] - L[j]) if L[j + 1] != L[j] else float(j)


def early_numbers(L, args):
    t0 = light_start(L)
    t = (np.arange(len(L)) - t0) * SAMPLE_NS
    total = L[(t >= -10) & (t <= args.total_us * 1000)].sum()
    fr = {w: L[(t >= -10) & (t <= w)].sum() / total for w in (10, 20, 50, 100)}
    p = int(np.argmax(L))
    sel = (np.arange(len(L)) >= p + 3) & (t <= args.early_ns)
    tau = np.nan
    try:
        x, y = t[sel], L[sel]
        f = lambda x, a, tau, c: a * np.exp(-(x - x[0]) / tau) + c
        popt, _ = curve_fit(f, x, y, p0=[y[0] - y[-1], 30.0, max(y[-1], 0)],
                            bounds=([0, 1, -np.inf], [np.inf, 500, np.inf]), maxfev=20000)
        tau = popt[1]
    except (RuntimeError, ValueError):
        pass
    return t, fr, tau


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.phases) as f:
        phases = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    blocks = [b for b in load(args.lifetime_dir, phases) if not args.only or b["label"] in args.only]
    regs = sorted(args.regs)
    mid = regs[len(regs) // 2]

    print(f"\n{'block':<22}{'phase':<8}{'<10 ns':>8}{'<20 ns':>8}{'<50 ns':>8}{'<100 ns':>9}"
          f"{'early tau (ns)':>16}")
    results = []
    for b in blocks:
        L = deconvolve(b["M"], b["R"], mid)
        t, fr, tau = early_numbers(L, args)
        results.append((b, t, L))
        print(f"{b['label']:<22}{b['phase']:<8}{fr[10]:>8.3f}{fr[20]:>8.3f}{fr[50]:>8.3f}{fr[100]:>9.3f}"
              f"{tau:>16.1f}")

    picks = [r for r in results if r[0]["label"] in args.examples] or results[:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 4.8), squeeze=False)
    for ax, (b, _, _) in zip(axes[0], picks):
        for k, reg in enumerate(regs):
            L = deconvolve(b["M"], b["R"], reg)
            t0 = light_start(L)
            t = (np.arange(len(L)) - t0) * SAMPLE_NS
            ax.plot(t, L / L.max(), color=PHASE_COLOR.get(b["phase"], "#555555"),
                    alpha=0.35 + 0.65 * k / max(len(regs) - 1, 1), linewidth=1.3, label=f"reg {reg:g}")
        ax.set_xlim(-30, args.early_ns)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.5)
        ax.set_xlabel("time from light start (ns)")
        ax.set_title(f"{b['label']} ({b['phase']}): light, R removed")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0][0].set_ylabel("light (peak = 1)")
    fig.tight_layout()
    fig.savefig(out / "early_light.png", dpi=120)
    plt.close(fig)

    # one panel: the three example phases on top of each other (middle reg)
    fig, ax = plt.subplots(figsize=(8, 5))
    for b, t, L in picks:
        ax.plot(t, L / L.max(), color=PHASE_COLOR.get(b["phase"], "#555555"), linewidth=1.5,
                label=f"{b['label']} ({b['phase']})")
    ax.set_xlim(-30, args.early_ns)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("time from light start (ns)")
    ax.set_ylabel("light (peak = 1)")
    ax.set_title(f"Early light by phase (reg {mid:g})")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "early_light_phases.png", dpi=120)
    plt.close(fig)
    print(f"\nplots in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifetime-dir", default="results/lifetime_m180")
    ap.add_argument("--output-dir", default="results/early_light")
    ap.add_argument("--phases", default="config/run_phases.yaml")
    ap.add_argument("--regs", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--early-ns", type=float, default=300)
    ap.add_argument("--total-us", type=float, default=7.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--examples", nargs="*",
                    default=["260520_1340:0-9", "260520_1800:70-79", "260520_1239:20-29"])
    main(ap.parse_args())
