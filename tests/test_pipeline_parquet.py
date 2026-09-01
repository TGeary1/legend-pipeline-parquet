# tests/test_pipeline_parquet.py
import pytest
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from lh5.io import store as lh5store

from pipeline.process import convert_to_raw, convert_to_dsp, compute_psd_params
from pipeline.process import process_file, select_channel_files
from pipeline.calibration import get_run_key


def _extract_struct_values(table, column_name):
    col = pc.struct_field(table.column(column_name), "values")
    return np.stack(col.to_numpy(zero_copy_only=False))


@pytest.mark.slow
def test_convert_to_raw_produces_expected_structure(test_daq_file, tmp_path):
    raw_path = convert_to_raw(test_daq_file, tmp_path, overwrite=True)
    assert raw_path is not None
    assert raw_path.exists()

    t = pq.read_table(str(raw_path))
    assert t.num_rows == 104_596

    expected_cols = {"board", "channel", "timestamp", "energy", "energy_short",
                      "waveform", "run", "cycle_id", "rownumber", "medium"}
    assert expected_cols.issubset(set(t.column_names))

    wf = _extract_struct_values(t, "waveform")
    assert wf.shape == (104_596, 5000)
    assert not np.isnan(wf.astype(float)).any()

    ts = t.column("timestamp").to_numpy()
    assert not (np.diff(ts) < 0).any()

    assert set(t.column("medium").to_pylist()) == {"liquid"}


@pytest.mark.slow
def test_convert_to_dsp_matches_lh5_baseline(test_daq_file, tmp_path, lh5_reference_hit_path):
    raw_path = convert_to_raw(test_daq_file, tmp_path, overwrite=True)
    dsp_path = convert_to_dsp(raw_path, tmp_path, "config/compass-dsp-config.json", overwrite=True)
    assert dsp_path is not None

    t = pq.read_table(str(dsp_path))
    assert t.num_rows == 104_596
    assert {"bl", "bl_sig", "wf_amplitude", "wf_charge_window"}.issubset(set(t.column_names))

    wcw = _extract_struct_values(t, "wf_charge_window")
    assert wcw.shape == (104_596, 4050)

    # Cross-validate against the independently-built LH5 dsp pipeline.
    # Note: lh5_reference_hit_path points at the hit tier, but bl/bl_sig are
    # carried through unchanged from dsp, so this is a valid check either way.
    lh5_hit = lh5store.LH5Store().read("CompassEvent", str(lh5_reference_hit_path))
    assert np.allclose(t.column("bl").to_numpy(), lh5_hit["bl"].nda)
    assert np.allclose(t.column("bl_sig").to_numpy(), lh5_hit["bl_sig"].nda)


@pytest.mark.slow
def test_compute_psd_params_matches_lh5_baseline(test_daq_file, tmp_path, lh5_reference_hit_path):
    raw_path = convert_to_raw(test_daq_file, tmp_path, overwrite=True)
    dsp_path = convert_to_dsp(raw_path, tmp_path, "config/compass-dsp-config.json", overwrite=True)
    hit_path = compute_psd_params(dsp_path, tmp_path, overwrite=True)
    assert hit_path is not None

    t = pq.read_table(str(hit_path))
    assert t.num_rows == 104_596

    psd_parquet = t.column("psd_param").to_numpy()
    lh5_hit = lh5store.LH5Store().read("CompassEvent", str(lh5_reference_hit_path))
    psd_lh5 = lh5_hit["psd_param"].nda

    assert np.isnan(psd_parquet).sum() == 538  # today's validated noise-floor count
    assert np.allclose(psd_parquet, psd_lh5, equal_nan=True)

    if "n_pe" in t.column_names and "n_pe" in lh5_hit.keys():
        assert np.allclose(t.column("n_pe").to_numpy(), lh5_hit["n_pe"].nda)


@pytest.mark.slow
def test_skip_if_exists_behavior(test_daq_file, tmp_path):
    raw_path_1 = convert_to_raw(test_daq_file, tmp_path, overwrite=True)
    mtime_1 = raw_path_1.stat().st_mtime

    raw_path_2 = convert_to_raw(test_daq_file, tmp_path, overwrite=False)
    mtime_2 = raw_path_2.stat().st_mtime

    assert mtime_1 == mtime_2


def test_get_run_key_matches_across_tiers():
    # Parquet filenames don't carry tier suffixes, unlike the LH5 pipeline's
    # _raw/_dsp/_hit convention — confirm the key still resolves consistently.
    raw_key = get_run_key("data/raw/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.parquet")
    dsp_key = get_run_key("data/dsp/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.parquet")
    hit_key = get_run_key("data/hit/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.parquet")
    assert raw_key == dsp_key == hit_key == "DataR_CH1@DT5730_1463_run_260520_1340_liquid_1"

@pytest.mark.slow
def test_process_file_end_to_end(test_daq_file, tmp_path):
    result = process_file(test_daq_file, base_dir=tmp_path, overwrite=True)
    assert result is not None
    assert result.exists()

    t = pq.read_table(str(result))
    assert t.num_rows == 104_596
    assert "psd_param" in t.column_names


def test_select_channel_files_filters_correctly():
    files = [
        "data/daq/DataR_CH0@DT5730_run1.BIN",
        "data/daq/DataR_CH1@DT5730_run1.BIN",
        "data/daq/DataR_CH2@DT5730_run1.BIN",
    ]
    result = select_channel_files(files, wanted_channels=[1])
    assert len(result) == 1
    assert "CH1@" in result[0]


def test_select_channel_files_handles_unparseable_names():
    files = ["data/daq/some_weird_filename.BIN"]
    result = select_channel_files(files, wanted_channels=[1])
    assert result == []