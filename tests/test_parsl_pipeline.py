# tests/test_parsl_pipeline.py
import pytest
import parsl
import numpy as np
import pyarrow.parquet as pq
from utils import make_local_config
from pipeline.parsl_apps import raw_stage_app, dsp_stage_app, hit_stage_app


@pytest.mark.slow
def test_parsl_pipeline_matches_direct_pipeline(test_daq_file, tmp_path):
    parsl.load(make_local_config())

    (tmp_path / "dsp").mkdir()
    (tmp_path / "hit").mkdir()

    try:
        raw_path = raw_stage_app(str(test_daq_file), str(tmp_path), overwrite=True).result()
        dsp_path = dsp_stage_app(raw_path, str(tmp_path / "dsp"), "config/compass-dsp-config.json", overwrite=True).result()
        hit_path = hit_stage_app(dsp_path, str(tmp_path / "hit"), "config/spe_calibration.yaml", overwrite=True).result()

        t = pq.read_table(hit_path)
        assert t.num_rows == 104_596
        assert np.isnan(t.column("psd_param").to_numpy()).sum() == 538
    finally:
        parsl.dfk().cleanup()