# tests/test_calibration.py
import numpy as np
from lh5.io import store as lh5store
from pipeline.calibration import fit_spe_calibration

def test_spe_calibration_matches_validated_result():
    tbl = lh5store.LH5Store().read("CompassEvent", "data/hit/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1_hit.lh5")
    charge_total = tbl["charge_total"].nda

    result = fit_spe_calibration(
        charge_total,
        peak_windows=[(1, 30000, 55000), (2, 78000, 98000), (3, 118000, 145000), (4, 160000, 195000)],
        hist_range=(0, 200000),
    )

    assert result["r_squared"] > 0.999
    assert 42500 < result["gain"] < 43700   # today's validated value ± margin
    assert result["gain_err"] < 500