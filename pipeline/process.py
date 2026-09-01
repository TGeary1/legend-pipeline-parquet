import logging
import warnings
import re
import lgdo
import numpy as np
import pyarrow as pa
from pipeline.calibration import get_run_key
from lh5.io import store as lh5store
from pathlib import Path
from daq2lh5 import build_raw
from dspeed import build_dsp
from .daq_config import COMPASS_CONFIG


log = logging.getLogger(__name__)

def convert_to_raw(daq_path, base_dir, overwrite=False):
    """base_dir should be the parent of a 'raw/' subdirectory that build_raw_app writes into."""
    from build_raw_compass import build_raw_app

    daq_path = Path(daq_path)
    expected_path = Path(base_dir) / "raw" / (daq_path.stem + ".parquet")

    if expected_path.exists() and not overwrite:
        log.info(f"raw SKIP (already exists): {expected_path.name}")
        return expected_path

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = build_raw_app(str(daq_path), str(base_dir))
            overflow_warnings = [w for w in caught if "overflow" in str(w.message).lower()]
            if overflow_warnings:
                log.warning(f"OVERFLOW WARNING during raw decode: {daq_path.name} — flag for manual verification")
    except Exception:
        log.exception(f"FAILED raw conversion: {daq_path}")
        return None

    log.info(f"raw OK: {daq_path.name} -> {Path(result).name}")
    return Path(result)


def convert_to_dsp(raw_path, dsp_dir, dsp_config, overwrite=False):
    import pyarrow.parquet as pq
    import pyarrow as pa
    import lgdo
    from dspeed import build_dsp

    raw_path = Path(raw_path)
    dsp_path = Path(dsp_dir) / raw_path.name

    if dsp_path.exists() and not overwrite:
        log.info(f"dsp SKIP (already exists): {dsp_path.name}")
        return dsp_path

    try:
        full_table = pq.read_table(str(raw_path))

        # Provenance columns aren't physics data — dspeed doesn't need them,
        # and lgdo.Array can't represent string columns like "medium" anyway.
        provenance_cols = ["run", "cycle_id", "rownumber", "medium"]
        physics_cols = [c for c in full_table.column_names if c not in provenance_cols]

        t_in = lgdo.Table(full_table.select(physics_cols))
        result = build_dsp(t_in, dsp_config=dsp_config)

        # Reattach provenance — row order is preserved by build_dsp
        out_arrow = (result if result is not None else t_in).view_as("arrow")
        for col in provenance_cols:
            out_arrow = out_arrow.append_column(col, full_table.column(col))

        pq.write_table(out_arrow, str(dsp_path), compression="lz4")
    except Exception:
        log.exception(f"FAILED dsp conversion: {raw_path}")
        return None

    log.info(f"dsp OK: {raw_path.name} -> {dsp_path.name}")
    return dsp_path


def process_file(daq_path, base_dir, dsp_config="config/compass-dsp-config.json",
                  calibration_config="config/spe_calibration.yaml", overwrite=False):
    daq_path = Path(daq_path)
    base_dir = Path(base_dir)

    for subdir in ("raw", "dsp", "hit"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    raw_path = convert_to_raw(daq_path, base_dir, overwrite=overwrite)
    if raw_path is None:
        return None

    dsp_path = convert_to_dsp(raw_path, base_dir / "dsp", dsp_config, overwrite=overwrite)
    if dsp_path is None:
        return None

    hit_path = compute_psd_params(dsp_path, base_dir / "hit", calibration_config, overwrite=overwrite)
    if hit_path is None:
        return None

    return hit_path


def select_channel_files(daq_files, wanted_channels):
    """Filter a list of DAQ file paths, keeping only the given CoMPASS channel numbers."""
    pattern = re.compile(r"CH(\d+)@")
    selected = []
    for f in daq_files:
        m = pattern.search(Path(f).name)
        if m is None:
            log.warning(f"Could not parse channel from filename, skipping: {f}")
            continue
        ch = int(m.group(1))
        if ch in wanted_channels:
            selected.append(f)
    return selected

import yaml
from pipeline.calibration import get_run_key

# pipeline/process.py (add this)

def compute_psd_params(dsp_path, hit_dir, calibration_config="config/spe_calibration.yaml", overwrite=False):
    """Compute charge_prompt, charge_total, psd_param (and n_pe, if calibrated)
    from a dsp-tier Parquet file's wf_charge_window column."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import yaml

    dsp_path = Path(dsp_path)
    hit_path = Path(hit_dir) / dsp_path.name

    if hit_path.exists() and not overwrite:
        log.info(f"hit SKIP (already exists): {hit_path.name}")
        return hit_path

    try:
        t = pq.read_table(str(dsp_path))

        wcw_col = pc.struct_field(t.column("wf_charge_window"), "values")
        wcw = np.stack(wcw_col.to_numpy(zero_copy_only=False))
        bl_sig = t.column("bl_sig").to_numpy()

        charge_prompt = wcw[:, :200].sum(axis=1)
        charge_total = wcw.sum(axis=1)

        n_window_samples = wcw.shape[1]
        noise_floor = 3 * bl_sig * np.sqrt(n_window_samples)
        psd_param = np.where(charge_total > noise_floor, charge_prompt / charge_total, np.nan)

        keep_cols = ["bl", "bl_sig", "wf_amplitude", "energy", "energy_short",
                     "run", "cycle_id", "rownumber", "medium"]
        out = t.select([c for c in keep_cols if c in t.column_names])
        out = (
            out.append_column("charge_prompt", pa.array(charge_prompt))
               .append_column("charge_total", pa.array(charge_total))
               .append_column("psd_param", pa.array(psd_param))
        )

        run_key = get_run_key(dsp_path)
        cal_path = Path(calibration_config)
        if cal_path.exists():
            with open(cal_path) as f:
                cal_db = yaml.safe_load(f) or {}
            if run_key in cal_db:
                gain = cal_db[run_key]["gain"]
                out = out.append_column("n_pe", pa.array(charge_total / gain))
                log.info(f"Applied SPE gain={gain:.1f} for {run_key}")
            else:
                log.warning(f"No SPE calibration found for {run_key} — n_pe omitted")
        else:
            log.warning(f"Calibration config {cal_path} not found — n_pe omitted")

        pq.write_table(out, str(hit_path), compression="lz4")
    except Exception:
        log.exception(f"FAILED hit-tier computation: {dsp_path}")
        return None

    log.info(f"hit OK: {dsp_path.name} -> {hit_path.name}")
    return hit_path