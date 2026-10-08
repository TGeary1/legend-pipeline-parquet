import numpy as np
from pipeline.process import max_rise, spe_pulse_height


def test_max_rise_detects_late_step():
    w = np.zeros((2, 1000), dtype=np.float32)
    w[1, 500:] = 650.0                      # a ~5 PE jump at sample 500 in event 1
    r = max_rise(w)
    assert r[0] == 0 and r[1] == 650


def test_max_rise_ignores_main_pulse_region():
    w = np.zeros((1, 1000), dtype=np.float32)
    w[0, 100:] = 1000.0                     # rise before JUMP_START: the main pulse, not pileup
    assert max_rise(w)[0] == 0


def test_spe_pulse_height():
    pulse = np.r_[np.zeros(10), 130.0, np.zeros(10)]
    wcw = np.tile(pulse, (2000, 1))
    n_pe = np.ones(2000)
    assert spe_pulse_height(wcw, n_pe) == 130.0
    assert spe_pulse_height(wcw[:10], n_pe[:10]) is None   # too few events
