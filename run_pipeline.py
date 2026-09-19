# run_pipeline.py
import argparse
import logging
from pathlib import Path
from pipeline.process import process_file, select_channel_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WANTED_CHANNELS = [1]  #Only channel [1] present within examined runs

def main(daq_dir, output_dir, limit=None):
    base_dir = Path(output_dir)
    for d in ("raw", "dsp", "hit"):
        (base_dir / d).mkdir(parents=True, exist_ok=True)

    all_daq_files = list(Path(daq_dir).rglob("*.BIN"))
    daq_files = select_channel_files(all_daq_files, WANTED_CHANNELS)
    logging.info(f"Found {len(all_daq_files)} DAQ files, selected {len(daq_files)} for channel(s) {WANTED_CHANNELS}")

    if limit is not None:
        daq_files = sorted(daq_files)[:limit]
        logging.info(f"Limiting to first {limit} file(s) for testing: {[Path(f).name for f in daq_files]}")

    results = {}
    for f in daq_files:
        results[f] = process_file(f, base_dir=base_dir, overwrite=False)

    succeeded = [f for f, r in results.items() if r is not None]
    failed = [f for f, r in results.items() if r is None]
    logging.info(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        logging.warning(f"Failed files: {[Path(f).name for f in failed]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daq-dir", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files in batch (for test purposes).")
    args = parser.parse_args()
    main(args.daq_dir, args.output_dir, limit=args.limit)
