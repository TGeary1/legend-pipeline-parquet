# tests/test_dsp_window_config.py
import pytest
from pipeline.dsp_config import build_dsp_config


def test_build_dsp_config_uses_correct_windows():
    cfg = build_dsp_config(baseline_end=450, prompt_end=650, total_end=4500)
    baseline_args = cfg["processors"]["bl, bl_sig, slope, intercept"]["args"]
    assert baseline_args[0] == "waveform[0:450]"

    charge_args = cfg["processors"]["wf_charge_window"]["args"]
    assert charge_args[0] == "wf_pol[450:4500]"


def test_build_dsp_config_scales_with_different_windows():
    cfg = build_dsp_config(baseline_end=480, prompt_end=680, total_end=3000)
    charge_args = cfg["processors"]["wf_charge_window"]["args"]
    assert charge_args[0] == "wf_pol[480:3000]"