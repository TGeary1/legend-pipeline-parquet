"""The three processing stages for one file: DAQ .BIN -> raw -> dsp -> hit.

Every stage writes Parquet, skips files whose output already exists (unless
overwrite=True), writes atomically (see atomic_write_table), and returns the
output path, or None on failure, so one bad file never stops a batch.
"""
import logging
import os
import re
import warnings
from pathlib import Path

import lgdo
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from dspeed import build_dsp

from build_raw_compass import parse_run_info
from pipeline.dsp_config import build_dsp_config
from pipeline.paths import DEFAULT_CALIBRATION, DEFAULT_WINDOW_CONFIG

log = logging.getLogger(__name__)

PROVENANCE_COLS = ["run", "cycle_id", "rownumber", "medium"]

# Pileup tag (see max_rise)
JUMP_LAG = 10      # 10 samples = 20 ns, about one photon's rise time
JUMP_START = 300   # 0.6 us into the charge window: past the main pulse peak


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


def _load_window_configs(window_config):
    with open(window_config) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stage 1: DAQ .BIN -> raw
# ---------------------------------------------------------------------------

def convert_to_raw(daq_path, base_dir, overwrite=False):
    """Decode one CoMPASS .BIN file into <base_dir>/raw/<stem>.parquet."""
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


# ---------------------------------------------------------------------------
# Stage 2: raw -> dsp
# ---------------------------------------------------------------------------

def convert_to_dsp(raw_path, dsp_dir, window_config=DEFAULT_WINDOW_CONFIG, overwrite=False):
    """Baseline, polarity-corrected charge window and scalars, via dspeed.
    The window boundaries come from window_config, keyed by the waveform
    length measured from the data itself."""
    raw_path = Path(raw_path)
    dsp_path = Path(dsp_dir) / raw_path.name
    Path(dsp_dir).mkdir(parents=True, exist_ok=True)

    if dsp_path.exists() and not overwrite:
        log.info(f"dsp SKIP (already exists): {dsp_path.name}")
        return dsp_path

    try:
        full_table = pq.read_table(str(raw_path))

        # The source of truth for the waveform length is the waveform array
        # itself, not the filename or the (known-unreliable) CompassHeader.
        wf_values = pc.struct_field(full_table.column("waveform"), "values")
        wf_len = len(wf_values.to_numpy(zero_copy_only=False)[0])

        configs = _load_window_configs(window_config)
        if wf_len not in configs:
            log.error(f"No dsp window config for wf_len={wf_len} in {window_config} — "
                      f"add an entry before processing this run. File: {raw_path.name}")
            return None

        cfg = configs[wf_len]
        log.info(f"Using dsp windows for wf_len={wf_len}: {cfg.get('notes', '')}")

        physics_cols = [c for c in full_table.column_names if c not in PROVENANCE_COLS]
        t_in = lgdo.Table(full_table.select(physics_cols))
        dsp_config = build_dsp_config(cfg["baseline_end"], cfg["prompt_end"], cfg["total_end"])
        result = build_dsp(t_in, dsp_config=dsp_config)

        out_arrow = (result if result is not None else t_in).view_as("arrow")
        for col in PROVENANCE_COLS:
            out_arrow = out_arrow.append_column(col, full_table.column(col))

        atomic_write_table(out_arrow, dsp_path, compression="lz4")
    except Exception:
        log.exception(f"FAILED dsp conversion: {raw_path}")
        return None

    log.info(f"dsp OK: {raw_path.name} -> {dsp_path.name}")
    return dsp_path


# ---------------------------------------------------------------------------
# Stage 3: dsp -> hit
# ---------------------------------------------------------------------------

def compute_psd_params(dsp_path, hit_dir, calibration_config=DEFAULT_CALIBRATION,
                       window_config=DEFAULT_WINDOW_CONFIG, overwrite=False, prompt_width=None):
    """Per-event physics quantities from a dsp file's wf_charge_window:

    always:          charge_prompt, charge_total, psd_param (prompt / total),
                     max_jump_adc (pileup tag), lowest_adc (saturation flag),
                     plus bl, bl_sig, wf_amplitude and the provenance columns
    if calibrated:   n_pe (charge_total / gain) and max_jump_pe (pileup tag
                     in units of this file's 1 PE pulse height)

    prompt_width (samples from the charge-window start) defaults to the
    value in window_config; pass it explicitly only to reproduce an older
    definition (the LH5 comparison test uses 200)."""
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

        # Find this file's window config the way convert_to_dsp chose it:
        # the charge-window length (total_end - baseline_end) is the one
        # quantity recoverable from the dsp file alone.
        wf_charge_len = wcw.shape[1]
        configs = _load_window_configs(window_config)
        matching = [c for c in configs.values() if c["total_end"] - c["baseline_end"] == wf_charge_len]
        if not matching:
            log.error(f"No window config matches wf_charge_window length {wf_charge_len} for {dsp_path.name}")
            return None
        if prompt_width is None:
            prompt_width = matching[0]["prompt_width"]

        charge_prompt = wcw[:, :prompt_width].sum(axis=1)
        charge_total = wcw.sum(axis=1)

        # Events whose total charge is within noise of zero get NaN PSD
        # rather than a meaningless ratio of two noise sums.
        noise_floor = 3 * bl_sig * np.sqrt(wf_charge_len)
        psd_param = np.where(charge_total > noise_floor, charge_prompt / charge_total, np.nan)

        max_jump_adc = max_rise(wcw)
        lowest_adc = t.column("bl").to_numpy() - t.column("wf_amplitude").to_numpy()

        # energy/energy_short are CoMPASS's on-board values; carried only if an
        # older dsp file still has them (the current dsp config drops them)
        keep_cols = ["bl", "bl_sig", "wf_amplitude", "energy", "energy_short"] + PROVENANCE_COLS
        out = t.select([c for c in keep_cols if c in t.column_names])
        out = (
            out.append_column("charge_prompt", pa.array(charge_prompt))
               .append_column("charge_total", pa.array(charge_total))
               .append_column("psd_param", pa.array(psd_param))
               .append_column("max_jump_adc", pa.array(max_jump_adc))
               .append_column("lowest_adc", pa.array(lowest_adc))
        )

        out = _add_calibrated_columns(out, dsp_path, Path(calibration_config),
                                      charge_total, wcw, max_jump_adc)

        atomic_write_table(out, hit_path, compression="lz4")
    except Exception:
        log.exception(f"FAILED hit-tier computation: {dsp_path}")
        return None

    log.info(f"hit OK: {dsp_path.name} -> {hit_path.name}")
    return hit_path


def _add_calibrated_columns(out, dsp_path, cal_path, charge_total, wcw, max_jump_adc):
    """Append n_pe and max_jump_pe if this run has an SPE calibration.
    Calibrations are keyed by run_base (from parse_run_info), so every file
    of a run shares one entry. A missing calibration is not an error: the
    columns are left out with a warning, and the hit tier can be rebuilt
    once the run is calibrated."""
    if not cal_path.exists():
        log.warning(f"Calibration config {cal_path} not found — n_pe omitted")
        return out

    with open(cal_path) as f:
        cal_db = yaml.safe_load(f) or {}
    bad_keys = [k for k in cal_db if not isinstance(k, str)]
    if bad_keys:
        log.warning(f"Non-string keys in {cal_path}: {bad_keys}. YAML read them as numbers; "
                    f"quote them, e.g. '260520_1340':")

    run_key = parse_run_info(dsp_path.stem)[0]
    if run_key not in cal_db:
        log.warning(f"No SPE calibration found for {run_key} — n_pe omitted")
        return out

    gain = cal_db[run_key]["gain"]
    n_pe = charge_total / gain
    out = out.append_column("n_pe", pa.array(n_pe))
    spe_peak = spe_pulse_height(wcw, n_pe)
    if spe_peak is not None:
        out = out.append_column("max_jump_pe", pa.array(max_jump_adc / spe_peak))
        log.info(f"1 PE pulse height {spe_peak:.1f} ADC for {dsp_path.name}")
    else:
        log.warning(f"too few 1 PE events in {dsp_path.name}; max_jump_pe omitted")
    log.info(f"Applied SPE gain={gain:.1f} for {run_key}")
    return out


# ---------------------------------------------------------------------------
# Helpers for batch drivers
# ---------------------------------------------------------------------------

def process_file(daq_path, base_dir, calibration_config=DEFAULT_CALIBRATION,
                 window_config=DEFAULT_WINDOW_CONFIG, overwrite=False):
    """All three stages for one file, sequentially. Returns the hit path or None."""
    base_dir = Path(base_dir)
    for subdir in ("raw", "dsp", "hit"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    raw_path = convert_to_raw(daq_path, base_dir, overwrite=overwrite)
    if raw_path is None:
        return None
    dsp_path = convert_to_dsp(raw_path, base_dir / "dsp", window_config, overwrite=overwrite)
    if dsp_path is None:
        return None
    return compute_psd_params(dsp_path, base_dir / "hit", calibration_config, window_config,
                              overwrite=overwrite)


def select_channel_files(daq_files, wanted_channels):
    """Filter a list of DAQ file paths, keeping only the given CoMPASS channel numbers."""
    pattern = re.compile(r"CH(\d+)@.*?(?:_(\d+))?\.BIN$", re.IGNORECASE)
    selected = []
    for f in daq_files:
        m = pattern.search(Path(f).name)
        if m is None:
            log.warning(f"Could not parse channel from filename, skipping: {f}")
            continue
        if int(m.group(1)) in wanted_channels:
            selected.append(f)
    return selected


def sequence_key(path):
    """Sort key putting a run's files in acquisition order: the unnumbered
    first file, then _1, _2, ... (plain sorted() would put _10 before _2).
    Uses parse_run_info so a date-time run name such as run_260520_1447 is
    not mistaken for file number 1447."""
    try:
        return parse_run_info(Path(path).stem)[1]
    except ValueError:
        return 0
