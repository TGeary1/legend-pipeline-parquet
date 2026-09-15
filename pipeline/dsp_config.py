# pipeline/dsp_config.py
def build_dsp_config(baseline_end, prompt_end, total_end):
    """Construct a dspeed processing chain dict with the given window
    boundaries, filled in programmatically rather than templated as text."""
    return {
        "outputs": ["bl", "bl_sig", "wf_amplitude", "wf_charge_window"],
        "processors": {
            "bl, bl_sig, slope, intercept": {
                "function": "linear_slope_fit",
                "module": "dspeed.processors",
                "args": [f"waveform[0:{baseline_end}]", "bl", "bl_sig", "slope", "intercept"],
                "unit": ["ADC", "ADC", "ADC", "ADC"],
            },
            "wf_blsub": {
                "function": "subtract", "module": "numpy",
                "args": ["waveform", "bl", "wf_blsub"], "unit": "ADC",
            },
            "wf_pol": {
                "function": "multiply", "module": "numpy",
                "args": ["wf_blsub", -1.0, "wf_pol"], "unit": "ADC",
            },
            "wf_amplitude": {
                "function": "amax", "module": "numpy",
                "args": ["wf_pol", 1, "wf_amplitude"],
                "kwargs": {"signature": "(n),()->()", "types": ["fi->f"]}, "unit": "ADC",
            },
            "wf_charge_window": {
                "function": "multiply", "module": "numpy",
                "args": [f"wf_pol[{baseline_end}:{total_end}]", 1.0, "wf_charge_window"],
                "unit": "ADC",
            },
        },
    }
