# legend-pipeline-parquet

Processing and analysis for argon scintillation data (gas, liquid and solid)
recorded with a CAEN DT5730 digitizer and CoMPASS. Each event is one PMT
waveform (2 ns samples). The pipeline turns CoMPASS `.BIN` files into three
Parquet tiers:

| Tier | Contents |
| --- | --- |
| raw | decoded waveforms, timestamps, and provenance (`run`, `cycle_id`, `rownumber`, `medium`) |
| dsp | baseline (`bl`, `bl_sig`), pulse height (`wf_amplitude`), baseline-subtracted charge window (`wf_charge_window`) |
| hit | `charge_prompt`, `charge_total`, `psd_param` (prompt / total), `max_jump_adc` (pileup tag), `lowest_adc` (saturation flag); plus `n_pe` and `max_jump_pe` once the run is calibrated |

[`PIPELINE_OVERVIEW_PARQUET.md`](PIPELINE_OVERVIEW_PARQUET.md) explains the
design decisions and the problems they solved. Read it before changing the
pipeline.

## Install

```bash
conda env create -f environment.yml        # on NERSC: module load conda first
conda activate legend-pipeline-parquet
python -m pytest                           # data-dependent tests skip on a fresh clone
```

Without conda: `pip install -r requirements.txt` (Python 3.10 or 3.11).
`requirements-lock.txt` is the full environment the May 20–21 results were produced with.

## Process a run

Locally, on a few files first (always stage new data):

```bash
python run_pipeline.py --daq-dir /path/to/DAQ/run_260520_1340_liquid --output-dir data_260520_1340_liquid --limit 2
```

On NERSC Perlmutter, from a login node inside `tmux` (Parsl submits the
Slurm jobs; one executor per stage):

```bash
python run_parsl_pipeline.py --daq-dir <run folder> --output-dir <output dir> --qos debug --limit 2
python run_parsl_pipeline.py --daq-dir <run folder> --output-dir <output dir> --qos regular \
    2>&1 | tee logs/<run>.log
```

Defaults: account `m2676`, conda environment `legend-pipeline-parquet`
(`--account`, `--conda-env` to change). Every stage skips files that already
exist, so rerunning the same command retries only what failed. The last file
of a run can be cut off by the DAQ stop and fail to decode; that is expected.

## Calibrate a run

Each run calibrates itself from its 1–4 photoelectron peaks:

```bash
python calibration/spe_search.py --hit-dir <run>/hit --label <name>     # find the peaks in the plots
python calibration/spe_fit.py --hit-dir <run>/hit --peak 1:LO:HI --peak 2:LO:HI \
    --peak 3:LO:HI --peak 4:LO:HI --dry-run                             # check gain, R^2 > 0.9999
python calibration/spe_fit.py ...same without --dry-run                 # writes config/spe_calibration.yaml
python run_pipeline.py --stage hit --run-dirs <run>                     # rebuild hit with n_pe
```

On NERSC, rebuild hit tiers on an interactive node (`salloc -N 1 -C cpu -q
interactive -t 01:00:00 -A m2676`) with `--workers 32`; each process needs
~10 GB.

## Analysis scripts

All run from the repo root and write plots under `results/`. Add each run's
phase to `config/run_phases.yaml` first.

| Script | What it shows |
| --- | --- |
| `analysis/calibration_plots.py` | standard per-run checks: baseline, PE peak fits, energy spectrum, pulse timing, 1 PE pulse, event rate |
| `analysis/stability_vs_time.py` | median PSD and alpha brightness per file across runs; **run this before trusting per-run numbers** |
| `analysis/phase_comparison.py` | PSD vs brightness by phase and by run, alpha peak fits and light yield |
| `analysis/psd_vs_npe.py` | quick PSD vs brightness comparison of a few runs |
| `analysis/forward_fit_analysis.py` | pulse-shape fit (singlet, intermediate, triplet) of a run's average waveform |
| `analysis/spe_deconvolution_analysis.py` | FFT deconvolution of the 1 PE response from the average waveform |
| `analysis/compare_run_settings.py` | side-by-side CoMPASS settings of several runs |

## Tests

`python -m pytest`. Tests that need data skip unless pointed at it:

| Variable | Enables |
| --- | --- |
| `PIPELINE_TEST_DAQ_FILE` | end-to-end tests; the expected row and NaN counts are for the liquid test file (default `data/daq/DataR_CH1@DT5730_1463_run_260520_1340_liquid_1.BIN`) |
| `PIPELINE_LH5_REFERENCE` | cross-validation against the older LH5 pipeline's output for that file |
| `SAR_FULL_RUN_HIT_DIR` | regression checks on the processed 1450pm SAr run |

`pytest -m "not slow"` skips the tests that decode a full file.

## Known limitations

- Waveform `dt` written by `daq2lh5` is a hard-coded 16 ns; the raw stage
  replaces it with the spacing from the run's `settings.xml` (2 ns).
- Only channel 1 is processed (`WANTED_CHANNELS` in the drivers).
- Hit files built before October 2026 used a 200-sample prompt window and
  are not comparable with current ones (280 samples, ~500 ns after onset).
