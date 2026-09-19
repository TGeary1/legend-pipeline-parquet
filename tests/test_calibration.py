# tests/test_calibration.py
import pytest
from pathlib import Path
import numpy as np
from lh5.io import store as lh5store
from pipeline.calibration import fit_spe_calibration


def test_spe_calibration_matches_validated_result(lh5_reference_hit_path):
    tbl = lh5store.LH5Store().read("CompassEvent", str(lh5_reference_hit_path))
    charge_total = tbl["charge_total"].nda

    result = fit_spe_calibration(
        charge_total,
        peak_windows=[(1, 30000, 55000), (2, 78000, 98000), (3, 118000, 145000), (4, 160000, 195000)],
        hist_range=(0, 200000),
    )

    assert result["r_squared"] > 0.999
    assert 42500 < result["gain"] < 43700
    assert result["gain_err"] < 500