"""Shared fixtures. Tests that need real data files skip (not fail) when the
file is missing, so a fresh clone gives "N passed, M skipped".

Point the data-dependent tests at your own copies with environment variables:
  PIPELINE_TEST_DAQ_FILE   a CoMPASS .BIN file (default: the liquid test file
                           under data/daq/)
  PIPELINE_LH5_REFERENCE   the LH5 pipeline's hit file for the same input
                           (default: ../legend-pipeline/data/hit/...)
"""
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DAQ_FILE = PROJECT_ROOT / "data/daq/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.BIN"
DEFAULT_LH5_REFERENCE = (PROJECT_ROOT.parent / "legend-pipeline" /
                         "data/hit/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1_hit.lh5")


@pytest.fixture
def test_daq_file():
    path = Path(os.environ.get("PIPELINE_TEST_DAQ_FILE", DEFAULT_DAQ_FILE))
    if not path.exists():
        pytest.skip(f"test DAQ file not found: {path} (set PIPELINE_TEST_DAQ_FILE)")
    return path


@pytest.fixture
def lh5_reference_hit_path():
    """The validated LH5 hit-tier output from the original (non-Parquet)
    pipeline, used to cross-validate this pipeline's numbers."""
    pytest.importorskip("lh5", reason="legend-lh5io not installed; LH5 cross-validation skipped")
    path = Path(os.environ.get("PIPELINE_LH5_REFERENCE", DEFAULT_LH5_REFERENCE))
    if not path.exists():
        pytest.skip(f"LH5 reference file not found: {path} (set PIPELINE_LH5_REFERENCE)")
    return path
