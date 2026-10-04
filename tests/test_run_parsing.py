# tests/test_run_parsing.py
from build_raw_compass import parse_run_info


def test_parse_run_info_date_time_with_medium():
    run_base, seq, medium = parse_run_info("DataR_CH1@DT5730_1463_run_260520_1340_liquid_1")
    assert run_base == "260520_1340"
    assert seq == 1
    assert medium == "liquid"


def test_parse_run_info_plain_numeric():
    run_base, seq, medium = parse_run_info("DataR_CH1@DT5730_1463_run_260521_1023_74")
    assert run_base == "260521_1023"
    assert seq == 74
    assert medium == "unknown"


def test_parse_run_info_label_style_unnumbered():
    run_base, seq, medium = parse_run_info("DataR_CH1@DT5730_1463_run_1450pm_SAr")
    assert run_base == "1450pm"
    assert seq == 0
    assert medium == "sar"


def test_parse_run_info_label_style_numbered():
    run_base, seq, medium = parse_run_info("DataR_CH1@DT5730_1463_run_1450pm_SAr_1")
    assert run_base == "1450pm"
    assert seq == 1
    assert medium == "sar"


def test_parse_run_info_raises_on_no_run_segment():
    import pytest
    with pytest.raises(ValueError):
        parse_run_info("SomeUnrelatedFilename")

def test_parse_run_info_gas():
    run_base, seq, medium = parse_run_info("DataR_CH1@DT5730_1463_run_260520_1239_gas_1")
    assert run_base == "260520_1239"
    assert seq == 1
    assert medium == "gas"

def test_parse_run_info_date_time_unnumbered():
    # the bug: trailing _1447 was read as sequence 1447
    assert parse_run_info("DataR_CH1@DT5730_1463_run_260520_1447") == ("260520_1447", 0, "unknown")

def test_parse_run_info_date_time_numbered():
    assert parse_run_info("DataR_CH1@DT5730_1463_run_260520_1447_5") == ("260520_1447", 5, "unknown")

def test_parse_run_info_gas_unnumbered():
    assert parse_run_info("DataR_CH1@DT5730_1463_run_260520_1239_gas") == ("260520_1239", 0, "gas")
