# regen_hit.py
"""Rebuild the hit tier of every run from its existing dsp files.
The old hit/ folder is kept as hit_prompt200/ until the new one is verified."""
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pipeline.process import compute_psd_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path("/global/cfs/cdirs/m2676/users/tgeary/legend-pipeline-parquet")
RUNS = ["data_260520_1239_gas", "data_260520_1340_liquid", "data_260520_1447",
        "data_260520_1448", "data_260520_1800", "data_260521_1023"]


def one(args):
    dsp_file, hit_dir = args
    return compute_psd_params(dsp_file, hit_dir) is not None


if __name__ == "__main__":
    jobs = []
    for run in RUNS:
        base = ROOT / run
        old, new = base / "hit", base / "hit_prompt200"
        if old.exists() and not new.exists():
            old.rename(new)                      # keep the old version, don't delete
        jobs += [(f, base / "hit") for f in sorted((base / "dsp").glob("*.parquet"))]

    # 32 processes x ~10 GB peak each fits in a 512 GB node, same limit as the dsp stage
    with ProcessPoolExecutor(max_workers=32) as pool:
        ok = list(pool.map(one, jobs))
    print(f"hit regenerated: {sum(ok)}/{len(jobs)} succeeded")
