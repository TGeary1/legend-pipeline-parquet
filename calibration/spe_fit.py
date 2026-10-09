"""
Step 3 of SPE calibration: fit the photoelectron peak ladder using windows
identified from spe_search.py's histograms, and write (or update) the
run's entry in config/spe_calibration.yaml.
 
The run key used in the YAML is derived from parse_run_info() (in
build_raw_compass.py), NOT the raw filename stem and NOT
pipeline.calibration.get_run_key() — those two differ meaningfully for
this project's Parquet files (see PIPELINE_OVERVIEW_PARQUET.md and the
process.py fix that keys compute_psd_params's calibration lookup by
run_base). Using the wrong key here means the calibration will silently
never be found when compute_psd_params runs — always let this script
derive the key rather than typing it in by hand.
 
Usage:
    python calibration/spe_fit.py \\
        --hit-dir /path/to/data/hit \\
        --n-files 5 \\
        --peak 1:20000:37000 \\
        --peak 2:48000:65000 \\
        --peak 3:75000:95000 \\
        --peak 4:105000:125000 \\
        --yaml-path config/spe_calibration.yaml
        # add --dry-run to see the result without writing to the YAML
"""
import argparse
import logging
from pathlib import Path

# Run as `python calibration/<script>.py` from the repo root: Python puts this
# script's folder on the import path, not the repo root, so add the root to
# reach the pipeline package and build_raw_compass.
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
 
import yaml
import numpy as np
 
from calibration.spe_search import load_charge_total  # noqa: E402
from pipeline.calibration import fit_spe_calibration  # noqa: E402
from build_raw_compass import parse_run_info  # noqa: E402
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
 
 
def parse_peak_arg(peak_str):
    """Parse a '--peak PE:LO:HI' argument into (pe_number, lo, hi)."""
    parts = peak_str.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--peak must be in the form PE:LO:HI (e.g. 1:20000:37000), got {peak_str!r}"
        )
    pe, lo, hi = parts
    return (int(pe), float(lo), float(hi))
 
 
def derive_run_key(files):
    """Derive the calibration lookup key from the first hit-tier file's
    name, using the same parse_run_info() logic compute_psd_params uses
    to look calibrations up. Warns if files from more than one run were
    accidentally aggregated together."""
    run_bases = {parse_run_info(Path(f).stem)[0] for f in files}
    if len(run_bases) > 1:
        log.warning(f"Files span more than one run_base ({run_bases}) — "
                    f"you are likely aggregating charge_total across different runs. "
                    f"Check --hit-dir points at a single run's output.")
    run_key = sorted(run_bases)[0]
    log.info(f"Derived calibration key: {run_key!r}")
    return run_key
 
 
# Written at the top of the YAML on every update (yaml.dump drops comments).
CALIBRATION_YAML_HEADER = """\
# SPE calibration per run, keyed by run_base from parse_run_info()
# (e.g. '260520_1340' for DataR_CH1@DT5730_1463_run_260520_1340_liquid_*.BIN).
# Written by calibration/spe_fit.py; read by the hit stage to add n_pe.
# Keys must be quoted strings: unquoted 260520_1340 is read by YAML as the
# number 2605201340 and the run's calibration is then never found.
# n_pe = charge_total / gain (the pedestal is recorded but not subtracted).
"""


def update_calibration_yaml(yaml_path, run_key, result, method_note):
    """Merge a new/updated calibration entry into the YAML file without
    disturbing any other runs' existing entries."""
    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        with open(yaml_path) as f:
            cal_db = yaml.safe_load(f) or {}
        bad = [k for k in cal_db if not isinstance(k, str)]
        if bad:
            log.warning(f"{yaml_path} has unquoted keys read as numbers: {bad}. "
                        f"They are kept as text but have lost their underscore; fix them by hand.")
            cal_db = {str(k): v for k, v in cal_db.items()}
    else:
        cal_db = {}
        log.warning(f"{yaml_path} did not exist — creating a new file")
 
    if run_key in cal_db:
        log.warning(f"Overwriting existing calibration entry for {run_key!r}")
 
    cal_db[run_key] = {
        "gain": round(float(result["gain"]), 1),
        "gain_err": round(float(result["gain_err"]), 1),
        "pedestal": round(float(result["pedestal"]), 1),
        "pedestal_err": round(float(result["pedestal_err"]), 1),
        "r_squared": round(float(result["r_squared"]), 6),
        "n_peaks_used": result["n_peaks_used"],
        "method": method_note,
    }
 
    with open(yaml_path, "w") as f:
        f.write(CALIBRATION_YAML_HEADER)
        yaml.dump(cal_db, f, default_flow_style=False, sort_keys=True)
 
    log.info(f"Wrote calibration for {run_key!r} to {yaml_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hit-dir", required=True, help="Directory of a single run's hit-tier parquet files")
    parser.add_argument("--n-files", type=int, default=5, help="Number of files to aggregate for the fit")
    parser.add_argument("--peak", action="append", type=parse_peak_arg, required=True,
                        help="A peak window as PE:LO:HI, e.g. --peak 1:20000:37000. "
                              "Repeat for each peak (at least 2 required, 4+ recommended).")
    parser.add_argument("--hist-bins", type=int, default=300)
    parser.add_argument("--yaml-path", default=str(REPO_ROOT / "config" / "spe_calibration.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Fit and print the result without writing to the YAML")
    args = parser.parse_args()
 
    if len(args.peak) < 2:
        parser.error("at least 2 --peak windows are required for a linear gain fit; 4+ is strongly recommended")
 
    charge_total, files = load_charge_total(args.hit_dir, args.n_files)
    run_key = derive_run_key(files)
 
    hist_range = (0, max(hi for _, _, hi in args.peak) * 1.2)
    result = fit_spe_calibration(charge_total, peak_windows=args.peak, bins=args.hist_bins, hist_range=hist_range)
 
    print(f"\n=== SPE calibration fit for {run_key!r} ===")
    print(f"gain       = {result['gain']:.1f} ± {result['gain_err']:.1f}")
    print(f"pedestal   = {result['pedestal']:.1f} ± {result['pedestal_err']:.1f}")
    print(f"R^2        = {result['r_squared']:.6f}")
    print(f"n_peaks    = {result['n_peaks_used']}")
 
    if result["r_squared"] < 0.999:
        log.warning(f"R^2={result['r_squared']:.6f} is lower than every calibration fit validated so far "
                    f"in this project (all were >0.9999) — double-check the peak windows against the "
                    f"search histogram before trusting this fit.")
 
    if args.dry_run:
        print("\n--dry-run given: not writing to the calibration YAML.")
    else:
        update_calibration_yaml(
            args.yaml_path, run_key, result,
            method_note="self-calibration from low-charge population, charge_total field",
        )
        print(f"\nWritten to {args.yaml_path}. Next: regenerate this run's hit-tier files "
              f"with compute_psd_params(..., overwrite=True) to populate n_pe.")
 
