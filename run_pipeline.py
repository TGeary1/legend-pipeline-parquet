"""
Local batch driver (no Parsl, no Slurm): run the pipeline on this machine,
optionally with several processes in parallel.

Full pipeline, DAQ files -> raw -> dsp -> hit:
    python run_pipeline.py --daq-dir data/daq --output-dir data [--limit 2] [--workers 4]

Hit stage only, rebuilt from existing dsp files (after a calibration or a
hit-stage code change; raw and dsp are untouched). On NERSC, run this on an
interactive compute node, not a login node:
    salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m2676
    python run_pipeline.py --stage hit --workers 32 --keep-previous hit_old \\
        --run-dirs $OUT/data_260520_1239_gas $OUT/data_260520_1340_liquid ...

--keep-previous NAME renames each run's existing hit/ to NAME/ before
rebuilding (skipped if NAME/ already exists), so the old results stay
available for comparison until you delete them.

Memory: each process needs ~10 GB for a 5000-sample file, so on a 512 GB
Perlmutter node keep --workers at 32 or below.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pipeline.paths import DEFAULT_CALIBRATION
from pipeline.process import compute_psd_params, process_file, select_channel_files, sequence_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WANTED_CHANNELS = [1]  # only channel present across all examined runs


def _full_job(args):
    daq_file, base_dir, calibration = args
    return process_file(daq_file, base_dir=base_dir, calibration_config=calibration) is not None


def _hit_job(args):
    dsp_file, hit_dir, calibration, overwrite = args
    return compute_psd_params(dsp_file, hit_dir, calibration, overwrite=overwrite) is not None


def _run(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs))


def run_full(daq_dir, output_dir, limit, workers, calibration):
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

    ok = _run(_full_job, [(f, base_dir, calibration) for f in daq_files], workers)
    failed = [Path(f).name for f, good in zip(daq_files, ok) if not good]
    logging.info(f"Done: {len(daq_files) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        logging.warning(f"Failed files: {failed}")


def run_hit_only(run_dirs, workers, calibration, keep_previous, overwrite):
    jobs = []
    for run_dir in map(Path, run_dirs):
        dsp_files = sorted((run_dir / "dsp").glob("*.parquet"))
        if not dsp_files:
            logging.warning(f"{run_dir}: no dsp files, skipped")
            continue
        hit_dir = run_dir / "hit"
        if keep_previous:
            old = run_dir / keep_previous
            if hit_dir.exists() and not old.exists():
                hit_dir.rename(old)
                logging.info(f"{run_dir.name}: previous hit/ kept as {keep_previous}/")
        jobs += [(f, hit_dir, calibration, overwrite) for f in dsp_files]

    ok = _run(_hit_job, jobs, workers)
    logging.info(f"hit rebuilt: {sum(ok)}/{len(jobs)} succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "hit"], default="all",
                        help="'all': DAQ -> raw -> dsp -> hit; 'hit': rebuild hit from existing dsp")
    parser.add_argument("--daq-dir", help="(--stage all) run folder containing .BIN files")
    parser.add_argument("--output-dir", default="data", help="(--stage all) raw/, dsp/, hit/ created inside")
    parser.add_argument("--limit", type=int, default=None, help="(--stage all) only the first N files")
    parser.add_argument("--run-dirs", nargs="+", help="(--stage hit) run output folders that contain dsp/")
    parser.add_argument("--keep-previous", default=None,
                        help="(--stage hit) rename existing hit/ to this name first, e.g. hit_old")
    parser.add_argument("--overwrite", action="store_true",
                        help="(--stage hit) replace existing hit files instead of skipping them")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="SPE calibration YAML")
    args = parser.parse_args()

    if args.stage == "all":
        if not args.daq_dir:
            parser.error("--stage all needs --daq-dir")
        run_full(args.daq_dir, args.output_dir, args.limit, args.workers, args.calibration)
    else:
        if not args.run_dirs:
            parser.error("--stage hit needs --run-dirs")
        run_hit_only(args.run_dirs, args.workers, args.calibration, args.keep_previous, args.overwrite)
