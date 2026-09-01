import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # tests/ -> project root

@pytest.fixture
def test_daq_file():
    path = Path("data/daq/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.BIN")
    if not path.exists():
        pytest.skip(f"Test Fixture Not Found: {path}")
    return path

@pytest.fixture
def lh5_reference_hit_path():
    """The validated LH5 hit-tier output from the original (non-Parquet) pipeline,
    used to cross-validate this pipeline's numerical output. Lives in a sibling
    repo — skip rather than fail if it's not present on this machine."""
    path = (
        Path(__file__).parent.parent.parent
        / "legend-pipeline"
        / "data/hit/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1_hit.lh5"
    )
    if not path.exists():
        pytest.skip(f"LH5 reference file not found: {path} — cross-validation skipped")
    return path