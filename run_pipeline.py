# run_pipeline.py
import logging
from pathlib import Path
from pipeline.process import process_file, select_channel_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WANTED_CHANNELS = [1]  # update to reflect your real physics channel(s)

if __name__ == "__main__":
    base_dir = Path("data")
    for d in ("raw", "dsp", "hit"):
        (base_dir / d).mkdir(parents=True, exist_ok=True)

    all_daq_files = list((base_dir / "daq").glob("*.BIN"))
    daq_files = select_channel_files(all_daq_files, WANTED_CHANNELS)
    logging.info(f"Found {len(all_daq_files)} DAQ files, selected {len(daq_files)} for channel(s) {WANTED_CHANNELS}")

    results = {}
    for f in daq_files:
        results[f] = process_file(f, base_dir=base_dir, overwrite=False)

    succeeded = [f for f, r in results.items() if r is not None]
    failed = [f for f, r in results.items() if r is None]
    logging.info(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        logging.warning(f"Failed files: {[Path(f).name for f in failed]}")