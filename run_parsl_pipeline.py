"""
Production batch driver: runs raw -> dsp -> hit for every selected DAQ file
in parallel via Parsl, on NERSC Perlmutter (Slurm) or locally.
 
Stages are chained with barriers: every raw conversion finishes before any
dsp starts, and every dsp finishes before any hit starts. Each stage runs on
its own Parsl executor (see pipeline/parsl_apps.py and utils.py).
 
Skip-if-exists is on (overwrite=False), so rerunning after a partial failure
only processes the files that did not finish.
 
Usage (from a NERSC login node, conda env active):
    python run_parsl_pipeline.py \\
        --daq-dir /global/cfs/cdirs/m2676/data/teststands/sarge/sarge9/DAQ/<run_folder> \\
        --output-dir /global/cfs/cdirs/m2676/users/<you>/legend-pipeline-parquet/<output_dir> \\
        --qos regular \\
        [--limit N]
 
Always stage new data: --limit 2, then --limit 10, then no limit.
"""
import argparse
import logging
from pathlib import Path
 
import parsl
 
from utils import make_config
from pipeline.parsl_apps import raw_stage_app, dsp_stage_app, hit_stage_app
from pipeline.process import select_channel_files
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 
WANTED_CHANNELS = [1]  # only channel present across all examined runs
 
 
def main(daq_dir, output_dir, account, qos, limit=None,
         calibration_config="config/spe_calibration.yaml"):
    config = make_config(account=account, qos=qos)
    parsl.load(config)
 
    try:
        base_dir = Path(output_dir)
        for d in ("raw", "dsp", "hit"):
            (base_dir / d).mkdir(parents=True, exist_ok=True)
 
        all_daq_files = list(Path(daq_dir).rglob("*.BIN"))
        daq_files = select_channel_files(all_daq_files, WANTED_CHANNELS)
        logging.info(f"Found {len(all_daq_files)} DAQ files, selected {len(daq_files)} "
                     f"for channel(s) {WANTED_CHANNELS}")
 
        if limit is not None:
            daq_files = sorted(daq_files)[:limit]
            logging.info(f"Limiting to first {limit} file(s): {[Path(f).name for f in daq_files]}")
 
        # Stage 1: raw (barrier: all finish before dsp starts)
        raw_futures = [raw_stage_app(str(f), str(base_dir), overwrite=False) for f in daq_files]
        raw_paths = [p for p in (f.result() for f in raw_futures) if p is not None]
        logging.info(f"raw: {len(raw_paths)}/{len(daq_files)} succeeded")
 
        # Stage 2: dsp
        dsp_futures = [dsp_stage_app(p, str(base_dir / "dsp"), overwrite=False) for p in raw_paths]
        dsp_paths = [p for p in (f.result() for f in dsp_futures) if p is not None]
        logging.info(f"dsp: {len(dsp_paths)}/{len(raw_paths)} succeeded")
 
        # Stage 3: hit
        hit_futures = [hit_stage_app(p, str(base_dir / "hit"), calibration_config, overwrite=False)
                       for p in dsp_paths]
        hit_paths = [p for p in (f.result() for f in hit_futures) if p is not None]
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
    parser.add_argument("--output-dir", default="data", help="Base output dir; raw/, dsp/, hit/ created inside")
    parser.add_argument("--account", default="m2676", help="NERSC charge account")
    parser.add_argument("--qos", default="debug",
                        help="Slurm QOS: 'debug' (30 min max) for staged tests, 'regular' for production")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N selected files")
    args = parser.parse_args()
    main(args.daq_dir, args.output_dir, args.account, args.qos, limit=args.limit)
