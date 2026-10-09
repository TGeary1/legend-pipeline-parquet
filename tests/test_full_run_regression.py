"""
Structural regression check for a full production run of run_1450pm_SAr,
locking in the shape of a known-good result (42 files, 7,270,088 total
events, ~14-38 NaN per file from the noise-floor cut) so a future pipeline
change that silently breaks something is caught without needing to rerun
the full 42-file batch on NERSC to notice.

This does NOT reprocess the run — it only reads existing hit-tier output.
Set SAR_FULL_RUN_HIT_DIR to that run's hit/ folder to enable it, e.g. on NERSC:
    SAR_FULL_RUN_HIT_DIR=/global/cfs/cdirs/m2676/users/tgeary/legend-pipeline-parquet/data_nersc_full/hit pytest
Without it the tests skip.
"""
import os

import pytest
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

FULL_RUN_HIT_DIR = Path(os.environ["SAR_FULL_RUN_HIT_DIR"]) if "SAR_FULL_RUN_HIT_DIR" in os.environ else None

EXPECTED_FILE_COUNT = 42
EXPECTED_TOTAL_ROWS = 7_271_088
STANDARD_CHUNK_ROWS = 174_037
NAN_COUNT_MIN_SANE = 0
NAN_COUNT_MAX_SANE = 100  # generous margin above the observed 14-38 range


@pytest.fixture
def full_run_hit_files():
    if FULL_RUN_HIT_DIR is None:
        pytest.skip("SAR_FULL_RUN_HIT_DIR not set — full-run regression check skipped")
    if not FULL_RUN_HIT_DIR.exists():
        pytest.skip(f"full-run output not found at {FULL_RUN_HIT_DIR}")
    files = sorted(FULL_RUN_HIT_DIR.glob("*.parquet"))
    if not files:
        pytest.skip(f"no .parquet files found in {FULL_RUN_HIT_DIR}")
    return files


def test_full_run_file_count(full_run_hit_files):
    assert len(full_run_hit_files) == EXPECTED_FILE_COUNT


def test_full_run_total_row_count(full_run_hit_files):
    total = sum(pq.read_table(str(f)).num_rows for f in full_run_hit_files)
    assert total == EXPECTED_TOTAL_ROWS


def test_full_run_row_counts_match_known_chunking(full_run_hit_files):
    """Every file should be the standard chunk size except exactly one
    shorter tail file (the final segment of the run)."""
    row_counts = [pq.read_table(str(f)).num_rows for f in full_run_hit_files]
    standard = [c for c in row_counts if c == STANDARD_CHUNK_ROWS]
    non_standard = [c for c in row_counts if c != STANDARD_CHUNK_ROWS]

    assert len(non_standard) == 1, (
        f"expected exactly one non-standard (tail) file, found {len(non_standard)}: {non_standard}"
    )
    assert len(standard) == EXPECTED_FILE_COUNT - 1


def test_full_run_nan_counts_are_sane(full_run_hit_files):
    """NaN counts (from the noise-floor cut) should stay in a physically
    sensible range for every file — no zeros (suspiciously perfect) and
    no large spikes (would suggest a processing problem in that file)."""
    for f in full_run_hit_files:
        t = pq.read_table(str(f))
        psd = t.column("psd_param").to_numpy()
        nan_count = np.isnan(psd).sum()
        assert NAN_COUNT_MIN_SANE < nan_count < NAN_COUNT_MAX_SANE, (
            f"{f.name}: NaN count {nan_count} outside sane range"
        )


def test_full_run_psd_param_distribution_stable(full_run_hit_files):
    """The bulk psd_param distribution should be consistent across every
    file in the run — a genuine outlier file would suggest a processing
    error (e.g. wrong dsp window applied) specific to that file."""
    medians = []
    for f in full_run_hit_files:
        t = pq.read_table(str(f))
        psd = t.column("psd_param").to_numpy()
        medians.append(np.nanmedian(psd))

    medians = np.array(medians)
    # all files' medians should cluster tightly — loose bound based on
    # observed run-wide consistency (~0.406), not a tight fit to noise
    assert medians.min() > 0.35
    assert medians.max() < 0.45
