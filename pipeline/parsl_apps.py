from parsl import python_app


@python_app(executors=["raw"])
def raw_stage_app(daq_path, base_dir, overwrite=False):
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    from pipeline.process import convert_to_raw
    result = convert_to_raw(daq_path, base_dir, overwrite=overwrite)
    return str(result) if result else None


@python_app(executors=["dsp"])
def dsp_stage_app(raw_path, dsp_dir, dsp_config, overwrite=False):
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    import numba
    numba.set_num_threads(1)

    from pipeline.process import convert_to_dsp
    result = convert_to_dsp(raw_path, dsp_dir, dsp_config, overwrite=overwrite)
    return str(result) if result else None


@python_app(executors=["hit"])
def hit_stage_app(dsp_path, hit_dir, calibration_config, overwrite=False):
    from pipeline.process import compute_psd_params
    result = compute_psd_params(dsp_path, hit_dir, calibration_config, overwrite=overwrite)
    return str(result) if result else None