import os
import logging
import warnings
import re
import lgdo
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
from pipeline.dsp_config import build_dsp_config
from build_raw_compass import parse_run_info
from lh5.io import store as lh5store
from pathlib import Path
from daq2lh5 import build_raw
from dspeed import build_dsp


log = logging.getLogger(__name__)

def atomic_write_table(table, path, **kwargs):
    """Write to '<name>.partial', then rename. A crash mid-write leaves only the
    .partial file, which skip-if-exists ignores, never a truncated .parquet."""
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    try:
        pq.write_table(table, str(tmp), **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

JUMP_LAG = 10      # 10 samples = 20 ns, about one photon's rise time
JUMP_START = 300   # 0.6 us into the charge window: past the main pulse peak


def max_rise(wcw, lag=JUMP_LAG, start=JUMP_START, chunk=5000):
    """Largest rise over any `lag` samples after `start`, per event, in ADC.
    Late argon light arrives one photon at a time (~1 PE steps); a sudden
    multi-PE rise is a second, unrelated event (pileup). Computed in chunks
    so the difference array never holds a whole file at once."""
    out = np.empty(len(wcw), dtype=np.float32)
    for i in range(0, len(wcw), chunk):
        w = wcw[i:i + chunk]
        out[i:i + chunk] = (w[:, start + lag:] - w[:, start:-lag]).max(axis=1)
    return out


def spe_pulse_height(wcw, n_pe, lo=0.85, hi=1.15, min_events=1000):
    """Peak height (ADC) of this file's average single-photoelectron pulse.
    Returns None if there are too few 1 PE events for a reliable average."""
    sel = (n_pe > lo) & (n_pe < hi)
    if sel.sum() < min_events:
        return None
    return float(wcw[sel].mean(axis=0).max())

#Runs the raw conversion app with overwrite/skip options and error isolation. 
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


#Measures waveform length from data and builds dsp
def convert_to_dsp(raw_path, dsp_dir, window_config="config/dsp_window_configs.yaml", overwrite=False):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import lgdo
    from dspeed import build_dsp

    raw_path = Path(raw_path)
    dsp_path = Path(dsp_dir) / raw_path.name

    Path(dsp_dir).mkdir(parents=True, exist_ok=True)

    if dsp_path.exists() and not overwrite:
        log.info(f"dsp SKIP (already exists): {dsp_path.name}")
        return dsp_path

    try:
        full_table = pq.read_table(str(raw_path))

        # Determine wf_len directly from the data, not from filenames or the
        # (known-unreliable) CompassHeader — the source of truth is the
        # actual waveform array length.
        wf_values = pc.struct_field(full_table.column("waveform"), "values")
        wf_len = len(wf_values.to_numpy(zero_copy_only=False)[0])

        with open(window_config) as f:
            configs = yaml.safe_load(f)
        if wf_len not in configs:
            log.error(f"No dsp window config for wf_len={wf_len} in {window_config} — "
                       f"add an entry before processing this run. File: {raw_path.name}")
            return None

        cfg = configs[wf_len]
        log.info(f"Using dsp windows for wf_len={wf_len}: {cfg.get('notes', '')}")

        provenance_cols = ["run", "cycle_id", "rownumber", "medium"]
        physics_cols = [c for c in full_table.column_names if c not in provenance_cols]

        t_in = lgdo.Table(full_table.select(physics_cols))
        dsp_config = build_dsp_config(cfg["baseline_end"], cfg["prompt_end"], cfg["total_end"])
        result = build_dsp(t_in, dsp_config=dsp_config)

        out_arrow = (result if result is not None else t_in).view_as("arrow")
        for col in provenance_cols:
            out_arrow = out_arrow.append_column(col, full_table.column(col))

        atomic_write_table(out_arrow, dsp_path, compression="lz4")
    except Exception:
        log.exception(f"FAILED dsp conversion: {raw_path}")
        return None

    log.info(f"dsp OK: {raw_path.name} -> {dsp_path.name}")
    return dsp_path


#Chains the three conversion stages for a single file
def process_file(daq_path, base_dir, dsp_config="config/compass-dsp-config.json",
                  calibration_config="config/spe_calibration.yaml", overwrite=False):
    daq_path = Path(daq_path)
    base_dir = Path(base_dir)

    for subdir in ("raw", "dsp", "hit"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    raw_path = convert_to_raw(daq_path, base_dir, overwrite=overwrite)
    if raw_path is None:
        return None

    dsp_path = convert_to_dsp(raw_path, base_dir / "dsp", overwrite=overwrite)
    if dsp_path is None:
        return None

    hit_path = compute_psd_params(dsp_path, base_dir / "hit", calibration_config, overwrite=overwrite)
    if hit_path is None:
        return None

    return hit_path


#Keeps only files for desired channels
def select_channel_files(daq_files, wanted_channels):
    """Filter a list of DAQ file paths, keeping only the given CoMPASS channel numbers."""
    pattern = re.compile(r"CH(\d+)@.*?(?:_(\d+))?\.BIN$", re.IGNORECASE)
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


def sequence_key(path):
    """Sort key so the unnumbered file sorts before _1, _2, _3, etc."""
    m = re.search(r"_(\d+)\.BIN$", Path(path).name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def compute_psd_params(dsp_path, hit_dir, calibration_config="config/spe_calibration.yaml", window_config="config/dsp_window_configs.yaml", overwrite=False, prompt_width=None):
    """Compute charge_prompt, charge_total, psd_param (and n_pe, if calibrated)
    from a dsp-tier Parquet file's wf_charge_window column."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import yaml

    dsp_path = Path(dsp_path)
    hit_path = Path(hit_dir) / dsp_path.name

    Path(hit_dir).mkdir(parents=True, exist_ok=True)

    if hit_path.exists() and not overwrite:
        log.info(f"hit SKIP (already exists): {hit_path.name}")
        return hit_path

    try:
        t = pq.read_table(str(dsp_path))

        wcw_col = pc.struct_field(t.column("wf_charge_window"), "values")
        wcw = np.stack(wcw_col.to_numpy(zero_copy_only=False))
        bl_sig = t.column("bl_sig").to_numpy()

        # Determine prompt_width the same way convert_to_dsp determined its
        # windows: keyed by the actual charge-window length, not assumed.
        wf_charge_len = wcw.shape[1]
        with open(window_config) as f:
            configs = yaml.safe_load(f)
        # Match by (total_end - baseline_end) == wf_charge_len, since that's
        # the quantity actually recoverable from this file alone.
        matching = [c for c in configs.values() if c["total_end"] - c["baseline_end"] == wf_charge_len]
        if not matching:
            log.error(f"No window config matches wf_charge_window length {wf_charge_len} for {dsp_path.name}")
            return None
        if prompt_width is None:
            prompt_width = matching[0]["prompt_width"]

        charge_prompt = wcw[:, :prompt_width].sum(axis=1)
        charge_total = wcw.sum(axis=1)

        n_window_samples = wcw.shape[1]
        noise_floor = 3 * bl_sig * np.sqrt(n_window_samples)
        psd_param = np.where(charge_total > noise_floor, charge_prompt / charge_total, np.nan)

        # pileup tag and saturation flag (computed here, attached to `out` below)
        max_jump_adc = max_rise(wcw)
        lowest_adc = t.column("bl").to_numpy() - t.column("wf_amplitude").to_numpy()

        keep_cols = ["bl", "bl_sig", "wf_amplitude", "energy", "energy_short",
                     "run", "cycle_id", "rownumber", "medium"]
        out = t.select([c for c in keep_cols if c in t.column_names])
        out = (
            out.append_column("charge_prompt", pa.array(charge_prompt))
               .append_column("charge_total", pa.array(charge_total))
               .append_column("psd_param", pa.array(psd_param))
               .append_column("max_jump_adc", pa.array(max_jump_adc))
               .append_column("lowest_adc", pa.array(lowest_adc))
        )

        run_base, _, _ = parse_run_info(Path(dsp_path).stem)
        run_key = run_base
        cal_path = Path(calibration_config)
        if cal_path.exists():
            with open(cal_path) as f:
                cal_db = yaml.safe_load(f) or {}
                bad_keys = [k for k in cal_db if not isinstance(k, str)]
                if bad_keys:
                    log.warning(f"Non-string keys in {cal_path}: {bad_keys}. YAML read them as numbers; "
                            f"quote them, e.g. '260520_1340':")
            if run_key in cal_db:
                gain = cal_db[run_key]["gain"]
                n_pe = charge_total / gain
                out = out.append_column("n_pe", pa.array(charge_total / gain))
                spe_peak = spe_pulse_height(wcw, n_pe)
                if spe_peak is not None:
                    out = out.append_column("max_jump_pe", pa.array(max_jump_adc / spe_peak))
                    log.info(f"1 PE pulse height {spe_peak:.1f} ADC for {dsp_path.name}")
                else:
                    log.warning(f"too few 1 PE events in {dsp_path.name}; max_jump_pe omitted") 
                log.info(f"Applied SPE gain={gain:.1f} for {run_key}")
            else:
                log.warning(f"No SPE calibration found for {run_key} — n_pe omitted")
        else:
            log.warning(f"Calibration config {cal_path} not found — n_pe omitted")

        atomic_write_table(out, hit_path, compression="lz4")
    except Exception:
        log.exception(f"FAILED hit-tier computation: {dsp_path}")
        return None

    log.info(f"hit OK: {dsp_path.name} -> {hit_path.name}")
    return hit_path
