"""
Step 2 of SPE calibration: aggregate charge_total across a run's hit-tier
files and produce histograms for visually identifying the photoelectron
peak ladder.
 
This does NOT fit anything — it only produces plots for you to inspect
and pick peak windows from, the same way every calibration in this
project has been done so far (liquid: 4 peaks in 30k-200k; SAr: 4 peaks
in 20k-125k). Peak positions and the right charge_total range are NOT
predictable in advance — they depend on gain, trigger threshold, and the
specific run's light yield. Always look at the plots before guessing
fit windows.
 
Usage:
    python spe_search.py \\
        --hit-dir /path/to/data/hit \\
        --output-dir results/spe_search \\
        --label my_new_run \\
        --n-files 5 \\
        --wide-max 200000
"""
import argparse
import logging
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
 
 
def load_charge_total(hit_dir, n_files=None):
    """Load and concatenate charge_total across (a subset of) a run's
    hit-tier files. n_files=None loads every file found; for a run with
    many files (e.g. the ~174k-event chunks seen in this project), 3-5
    files is already ample statistics for a calibration fit."""
    hit_dir = Path(hit_dir)
    files = sorted(hit_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no .parquet files found in {hit_dir}")
 
    if n_files is not None:
        files = files[:n_files]
 
    log.info(f"Loading charge_total from {len(files)} file(s) in {hit_dir}")
    charge_total = np.concatenate([
        pq.read_table(str(f)).column("charge_total").to_numpy() for f in files
    ])
    log.info(f"{len(charge_total)} events loaded")
    return charge_total, files
 
 
def print_percentiles(charge_total):
    """Always look at the real scale before choosing a histogram range —
    this project has twice found the low-charge population sitting at a
    very different scale than initially assumed."""
    valid = charge_total[charge_total > 0]
    percentiles = [1, 10, 25, 50, 75, 99]
    values = np.percentile(valid, percentiles)
    log.info("charge_total percentiles:")
    for p, v in zip(percentiles, values):
        log.info(f"  {p}th: {v:.1f}")
    return dict(zip(percentiles, values))
 
 
def plot_search_histograms(charge_total, output_dir, label, low_max=None, wide_max=200000, bins=300):
    """Produce two histograms: a low-range zoom (auto-scaled from the
    25th percentile unless low_max is given) and a wide-range view to
    see the full peak ladder and any higher-energy tail structure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    valid = charge_total[(charge_total > 0)]
 
    if low_max is None:
        # a generous default: a few times the 25th percentile, so the
        # first peak or two is very likely visible without needing to
        # guess a range by hand first
        low_max = 3 * np.percentile(valid, 25)
        log.info(f"low_max not given, defaulting to {low_max:.0f} (3x the 25th percentile)")
 
    plt.figure()
    plt.hist(valid[valid < low_max], bins=bins)
    plt.xlabel("charge_total")
    plt.ylabel("count")
    plt.title(f"{label}: low-range charge_total")
    low_path = output_dir / f"{label}_spe_search_low.png"
    plt.savefig(low_path)
    plt.close()
 
    plt.figure()
    plt.hist(valid[valid < wide_max], bins=bins)
    plt.xlabel("charge_total")
    plt.ylabel("count")
    plt.title(f"{label}: wide-range charge_total")
    wide_path = output_dir / f"{label}_spe_search_wide.png"
    plt.savefig(wide_path)
    plt.close()
 
    log.info(f"Saved {low_path} and {wide_path}")
    return low_path, wide_path
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hit-dir", required=True, help="Directory of a run's hit-tier parquet files")
    parser.add_argument("--output-dir", default="results/spe_search", help="Where to save the histogram plots")
    parser.add_argument("--label", required=True, help="Short label for this run, used in filenames/titles")
    parser.add_argument("--n-files", type=int, default=5, help="Number of files to aggregate (default 5; use a smaller "
                                                                  "number for a quick look, more for better statistics)")
    parser.add_argument("--low-max", type=float, default=None, help="Upper bound for the low-range histogram "
                                                                       "(default: auto, 3x the 25th percentile)")
    parser.add_argument("--wide-max", type=float, default=200000, help="Upper bound for the wide-range histogram")
    args = parser.parse_args()
 
    charge_total, files = load_charge_total(args.hit_dir, args.n_files)
    print_percentiles(charge_total)
    plot_search_histograms(charge_total, args.output_dir, args.label, args.low_max, args.wide_max)
 
    print("\nNext: inspect the two saved plots, identify the evenly-spaced photoelectron "
          "peak ladder, and pass the corresponding (pe_number, lo, hi) windows to spe_fit.py")
