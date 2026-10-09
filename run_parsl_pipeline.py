"""
Production batch driver: runs raw -> dsp -> hit for every selected DAQ file
in parallel via Parsl, on NERSC Perlmutter (Slurm).

Stages are chained with barriers: every raw conversion finishes before any
dsp starts, and every dsp finishes before any hit starts. Each stage runs on
its own Parsl executor with its own worker limit (pipeline/parsl_config.py).

Skip-if-exists is on, so rerunning after a partial failure only processes
the files that did not finish. Failed tasks are logged as
"<stage> task failed: ..." and the run carries on.

Usage (from a NERSC login node, conda env active, inside tmux):
    python run_parsl_pipeline.py \\
        --daq-dir /global/cfs/cdirs/m2676/data/teststands/sarge/sarge9/DAQ/<run_folder> \\
        --output-dir $SCRATCH/<run_name> \\
        --qos regular \\
        [--limit N]

Always stage new data: --limit 2 on --qos debug first, then the full run.
To rebuild only the hit tier from existing dsp files, use
`python run_pipeline.py --stage hit` on an interactive node instead.
"""
import argparse
import logging
from pathlib import Path

import parsl

from pipeline.parsl_apps import dsp_stage_app, hit_stage_app, raw_stage_app
from pipeline.parsl_config import DEFAULT_ACCOUNT, DEFAULT_CONDA_ENV, make_config
from pipeline.paths import DEFAULT_CALIBRATION
from pipeline.process import select_channel_files, sequence_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("parsl").setLevel(logging.INFO)

WANTED_CHANNELS = [1]  # only channel present across all examined runs


def gather(futures, stage):
    """Collect results, logging failed tasks instead of crashing the whole run.
    A crash would trigger cleanup, which cancels every worker mid-write."""
    paths = []
    for f in futures:
        try:
            p = f.result()
        except Exception as e:
            logging.error(f"{stage} task failed: {type(e).__name__}: {e}")
            continue
        if p is not None:
            paths.append(p)
    return paths


def main(daq_dir, output_dir, account=DEFAULT_ACCOUNT, qos="debug", limit=None, walltime=None,
         conda_env=DEFAULT_CONDA_ENV, calibration_config=DEFAULT_CALIBRATION):
    config = make_config(account=account, qos=qos, walltime=walltime, conda_env=conda_env)
    parsl.load(config)

    try:
        base_dir = Path(output_dir)
        for d in ("raw", "dsp", "hit"):
            (base_dir / d).mkdir(parents=True, exist_ok=True)

        all_daq_files = list(Path(daq_dir).rglob("*.BIN"))
        daq_files = sorted(select_channel_files(all_daq_files, WANTED_CHANNELS), key=sequence_key)
        logging.info(f"Found {len(all_daq_files)} DAQ files, selected {len(daq_files)} "
                     f"for channel(s) {WANTED_CHANNELS}")

        if limit is not None:
            daq_files = daq_files[:limit]
            logging.info(f"Limiting to first {limit} file(s): {[Path(f).name for f in daq_files]}")

        # Stage 1: raw (barrier: all finish before dsp starts)
        raw_futures = [raw_stage_app(str(f), str(base_dir), overwrite=False) for f in daq_files]
        raw_paths = gather(raw_futures, "raw")
        logging.info(f"raw: {len(raw_paths)}/{len(daq_files)} succeeded")

        # Stage 2: dsp
        dsp_futures = [dsp_stage_app(p, str(base_dir / "dsp"), overwrite=False) for p in raw_paths]
        dsp_paths = gather(dsp_futures, "dsp")
        logging.info(f"dsp: {len(dsp_paths)}/{len(raw_paths)} succeeded")

        # Stage 3: hit
        hit_futures = [hit_stage_app(p, str(base_dir / "hit"), str(calibration_config), overwrite=False)
                       for p in dsp_paths]
        hit_paths = gather(hit_futures, "hit")
        logging.info(f"hit: {len(hit_paths)}/{len(dsp_paths)} succeeded")

        n_failed = len(daq_files) - len(hit_paths)
        if n_failed:
            logging.warning(f"{n_failed} file(s) did not complete all three stages — "
                            f"rerun the same command to retry only those (skip-if-exists is on)")
    finally:
        parsl.dfk().cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--daq-dir", required=True, help="Run folder containing .BIN files (searched recursively)")
    parser.add_argument("--output-dir", required=True, help="Base output dir; raw/, dsp/, hit/ created inside")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help="NERSC charge account")
    parser.add_argument("--qos", default="debug",
                        help="Slurm QOS: 'debug' (30 min max) for staged tests, 'regular' for production")
    parser.add_argument("--walltime", default=None,
                        help="Slurm walltime HH:MM:SS (default: 30 min for debug, 2 h for regular)")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV,
                        help="conda environment the compute-node workers activate")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="SPE calibration YAML")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files of the run")
    args = parser.parse_args()
    main(args.daq_dir, args.output_dir, account=args.account, qos=args.qos, limit=args.limit,
         walltime=args.walltime, conda_env=args.conda_env, calibration_config=args.calibration)
