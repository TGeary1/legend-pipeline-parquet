"""
Pulse shape discrimination: prompt/total light (psd_param) versus event
brightness (n_pe), compared across runs.

Different particle types produce different singlet/triplet mixes. Densely
ionizing particles (alphas) give more prompt light, so they should form a
band at higher psd_param than gammas/electrons. Bands only become distinct
at higher n_pe: with few photons, the prompt fraction is dominated by
random fluctuations (roughly 1/sqrt(n_pe)) and by the single-photon pulse
shape.

Reads only the n_pe and psd_param columns of each hit file, so it is light
enough for a login node.

Usage (from the repo root):
    python analysis/psd_vs_npe.py \\
        --run-dirs <data_dir_1> <data_dir_2> \\
        --labels gas liquid \\
        --output-dir results/psd_vs_npe

Note: psd_param currently uses the prompt_width in dsp_window_configs.yaml
(200 samples from the charge-window start, about 340 ns after pulse onset).
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NPE_BANDS = [(1, 3), (3, 10), (10, 30), (30, 100), (100, 500)]


def load(run_dir):
    files = sorted((Path(run_dir) / "hit").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no hit files in {run_dir}/hit")
    tables = [pq.read_table(str(f), columns=["n_pe", "psd_param"]) for f in files]
    n_pe = np.concatenate([t.column("n_pe").to_numpy() for t in tables])
    psd = np.concatenate([t.column("psd_param").to_numpy() for t in tables])
    ok = np.isfinite(psd) & np.isfinite(n_pe) & (n_pe > 0)   # drop noise-floor NaNs
    return n_pe[ok], psd[ok], len(files)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {label: load(d) for label, d in zip(args.labels, args.run_dirs)}

    x_bins = np.logspace(np.log10(0.5), np.log10(500), 120)   # log-spaced brightness bins
    y_bins = np.linspace(-0.1, 1.1, 120)

    # --- 2D histograms, one panel per run ---
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), sharey=True, squeeze=False)
    for ax, (label, (n_pe, psd, nf)) in zip(axes[0], data.items()):
        h = ax.hist2d(n_pe, psd, bins=[x_bins, y_bins], norm=LogNorm(), cmap="viridis")
        ax.set_xscale("log")
        ax.set_xlabel("n_pe (photoelectrons)")
        ax.set_title(f"{label} ({nf} files, {len(n_pe):,} events)")
        fig.colorbar(h[3], ax=ax, label="events")
    axes[0][0].set_ylabel("psd_param (prompt / total)")
    plt.tight_layout()
    plt.savefig(out / "psd_vs_npe_2d.png", dpi=120)
    plt.close()

    # --- PSD distribution in brightness bands, runs overlaid ---
    fig, axes = plt.subplots(1, len(NPE_BANDS), figsize=(4 * len(NPE_BANDS), 4), sharey=False)
    for ax, (lo, hi) in zip(axes, NPE_BANDS):
        for label, (n_pe, psd, _) in data.items():
            sel = (n_pe >= lo) & (n_pe < hi)
            if sel.sum() > 0:
                ax.hist(psd[sel], bins=y_bins, density=True, histtype="step", label=f"{label} ({sel.sum():,})")
        ax.set_title(f"{lo} ≤ n_pe < {hi}")
        ax.set_xlabel("psd_param")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("normalized events")
    plt.tight_layout()
    plt.savefig(out / "psd_in_npe_bands.png", dpi=120)
    plt.close()

    # --- summary numbers ---
    print(f"{'run':<10}" + "".join(f"{f'{lo}-{hi} PE':>16}" for lo, hi in NPE_BANDS))
    for label, (n_pe, psd, _) in data.items():
        row = []
        for lo, hi in NPE_BANDS:
            sel = (n_pe >= lo) & (n_pe < hi)
            row.append(f"{np.median(psd[sel]):.3f} ({sel.mean():.1%})" if sel.any() else "-")
        print(f"{label:<10}" + "".join(f"{r:>16}" for r in row))
    print("\neach cell: median psd_param in that brightness band (share of the run's events)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, help="processed run directories (containing hit/)")
    p.add_argument("--labels", nargs="+", required=True, help="one label per run dir")
    p.add_argument("--output-dir", default="results/psd_vs_npe")
    args = p.parse_args()
    if len(args.labels) != len(args.run_dirs):
        p.error("give exactly one --labels entry per --run-dirs entry")
    main(args)
