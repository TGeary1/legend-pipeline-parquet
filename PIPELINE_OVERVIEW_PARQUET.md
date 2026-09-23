# Solid Argon Scintillation Pipeline — Parquet Edition

This document covers `legend-pipeline-parquet`, the Parquet/pyarrow-based
sibling of the original LH5 pipeline (`legend-pipeline`,
see `PIPELINE_OVERVIEW.md` there). Both implement the same DAQ → raw → dsp →
hit physics chain for CoMPASS binary data; this repo exists because the
group is moving toward Parquet as the standard storage format, for faster
columnar analysis and better interoperability with the broader Python
data-science ecosystem than LH5/HDF5 offers.

**If you're new to this repo:** read this document fully before making
changes. Several pieces of this pipeline were generalized under real
pressure from genuinely new data (different run-naming conventions,
different waveform lengths) discovered partway through the project, and
the reasoning behind each generalization matters for using the pipeline
correctly on the *next* new run you encounter.

## 1. The big picture — same three tiers as the LH5 pipeline

```
DAQ binary (.BIN)
      │  build_raw_compass.py (open_stream-based, not daq2lh5.build_raw)
      ▼
raw tier (.parquet)      — decoded waveforms + metadata + provenance columns
      │  dspeed.build_dsp() called on an in-memory lgdo.Table
      ▼
dsp tier (.parquet)       — baseline-subtracted, polarity-corrected waveform
      │                      slice ("wf_charge_window") + scalar quantities
      │  numpy (compute_psd_params)
      ▼
hit tier (.parquet)       — charge_prompt, charge_total, psd_param, n_pe
                             (if calibrated) + carried-forward quantities
```

Every stage was **cross-validated numerically** against the already-trusted
LH5 pipeline's output on the same input file (`np.allclose` on `bl`,
`bl_sig`, `wf_charge_window`, `psd_param`, and `n_pe`) before being trusted.
See `tests/test_pipeline_parquet.py` — the `*_matches_lh5_baseline` tests
are the actual proof this reimplementation is correct, not just structurally
similar.

## 2. Why this repo uses `open_stream()` instead of `daq2lh5.build_raw()`

The LH5 pipeline uses `daq2lh5.build_raw()`, a high-level convenience
function. This repo's raw-conversion script (`build_raw_compass.py`,
originally written by a colleague for a different detector/experiment,
adapted here for CoMPASS) uses `daq2lh5.open_stream()` directly — a
lower-level streaming API — because it needs fine control over the output
schema: writing directly to Parquet with custom per-column encoding
(`DELTA_BINARY_PACKED` for waveform samples, which compresses well since
consecutive ADC samples are usually close in value) and stamping
provenance columns (`run`, `cycle_id`, `rownumber`, `medium`) onto every
row as it's written, rather than as a separate post-processing pass.

## 3. Provenance columns

Every row carries:
- **`run`** — a string identifier for the run (see §4 — this had to become
  a string, not an integer, once non-numeric run-naming conventions
  appeared)
- **`cycle_id`** — the sequence number of this file within the run (0 for
  the first/unnumbered file, 1, 2, 3... for continuations)
- **`rownumber`** — position within *this file* (resets to 0 per file, does
  not accumulate across a run's multiple files)
- **`medium`** — `"liquid"`, `"solid"`/`"sar"`/`"lar"`, or `"unknown"` if
  the filename doesn't encode it (see §4)

These were added specifically so that once multiple runs' hit-tier outputs
get concatenated for analysis, every row can still be traced back to its
source run without external bookkeeping — adopted from a colleague's
reference pipeline pattern.

## 4. Run-name parsing (`build_raw_compass.py::parse_run_info`)

**This had to be generalized after real data broke the original
assumption**, and it's worth understanding why, since a fourth naming
convention may well show up in future data.

CoMPASS filenames follow the pattern
`DataR_CH<n>@<board>_<serial>_run_<tag>_<seq>.BIN`, but `<tag>` has turned
out to vary a lot across runs actually collected:

| Example | `run_base` | `seq` | `medium` |
|---|---|---|---|
| `..._run_260520_1340_liquid_1` | `260520_1340` | 1 | `liquid` |
| `..._run_260521_1023_74` | `260521_1023` | 74 | `unknown` (no medium in name) |
| `..._run_1450pm_SAr` | `1450pm` | 0 | `sar` |
| `..._run_1450pm_SAr_1` | `1450pm` | 1 | `sar` |

The original implementation hardcoded a single regex
(`_run_(\d+)_(\d+)_(\w+)_(\d+)$`) that only matched the first pattern. It
broke on real data twice — once for the plain-numeric convention, once for
the am/pm-labeled convention — before being replaced with the current
approach: treat everything after `_run_` as the run tag, strip a trailing
bare integer as the sequence number (absence of one means sequence 0 —
**the first/unnumbered file of a run, not a file to exclude** — see §5),
then best-effort search the remaining tag for a known medium keyword.

**`medium == "unknown"` is expected and correct for the plain-numeric
naming convention** — those filenames genuinely don't encode a medium.
Before trusting analysis that groups by medium, confirm with the group
what medium those specific runs actually correspond to; don't assume.

**If a new run shows a different naming pattern that this logic still gets
wrong**, don't patch in a fourth special-cased regex reactively — the
current approach was specifically designed to be structural rather than
one-pattern-per-convention. Extend `parse_run_info()`'s logic (e.g. the
medium-keyword list) rather than adding another `if`/`elif` branch tied to
one specific run.

## 5. The unnumbered file is real data, not a header-only file

Early in this pipeline's development, a colleague's description of "the
unnumbered file contains header information" led to `select_channel_files`
briefly **excluding** files with no trailing sequence number. This was
wrong: the unnumbered file is genuinely the **first segment of a run's
acquisition**, containing real waveform events (confirmed directly by
decoding one and inspecting it — tens of thousands of real `CompassEvent`
rows, not just header metadata), continued by the numbered files that
follow it.

`select_channel_files` now includes the unnumbered file, and
`sequence_key()` sorts it first (as sequence 0) ahead of `_1`, `_2`, etc.,
so any downstream analysis that depends on event ordering across a run's
multiple files gets them in the correct order.

CoMPASS's actual per-file **header** (via `CompassHeaderDecoder`) was
separately investigated and found to be almost entirely unpopulated
(`nan` for `sample_rate`, `adcBitCount`, per-channel enable flags) except
for `wf_len` — see §6. Trigger thresholds, PSD gate widths, and PMT HV are
**not** recoverable from this header; if that acquisition metadata is ever
needed, check for a CoMPASS XML settings sidecar file instead, or ask
whoever ran the acquisition directly.

## 6. Waveform length varies by run — the dsp stage generalization

The very first validated run (`260520_1340_liquid_1`) has 5000-sample
waveforms. A later run (`1450pm_SAr`) turned out to have only 3000-sample
waveforms — confirmed both from `CompassHeaderDecoder`'s `wf_len` field and
from the actual decoded waveform array shape. Every dsp-stage window
boundary (baseline region, prompt window, total-charge window) had been
chosen as **fixed sample indices** tuned specifically for the 5000-sample
record; naively applying them to a 3000-sample record would either error
on an out-of-range slice or silently produce a wrong, truncated
`wf_charge_window`.

**The fix: `config/dsp_window_configs.yaml`, keyed by waveform length**,
determined from the *data itself* (not the unreliable header, not the
filename):

```yaml
5000:
  baseline_end: 450
  prompt_end: 650
  prompt_width: 200
  total_end: 4500
  notes: "liquid argon, validated against run 260520_1340_liquid_1"

3000:
  baseline_end: 480
  prompt_end: 680
  prompt_width: 200
  total_end: 3000
  notes: "solid argon, run 1450pm — decay not fully settled by end of record; total_end uses full record as a stopgap"
```

`convert_to_dsp()` reads the actual waveform array's length from the raw
Parquet file, looks up the matching entry, and **fails loudly (returns
`None`, logs an error) if no entry exists for that length** — it does not
guess, scale proportionally, or fall back to a default. This is
deliberate: a silently-wrong window choice would be far harder to catch
than an explicit "add a config entry for this" error. If you see this
error for a genuinely new run, don't just add a config entry — first plot
that run's average waveform (baseline-subtracted, polarity-corrected, the
same way §7's caveat was discovered) and pick real boundaries from it,
following the same process documented in the git history / conversation
that produced the two entries above. Do not assume proportional scaling
from an existing entry works — it demonstrably didn't (see §7).

`compute_psd_params()` independently looks up `prompt_width` for the
`wf_charge_window` length it receives, matched by
`total_end - baseline_end == len(wf_charge_window)`, so it stays correct
without needing the original absolute sample positions.

`pipeline/dsp_config.py::build_dsp_config()` constructs the `dspeed`
processing chain dict programmatically from these three numbers, rather
than templating a static JSON file — this was itself a fix: earlier
attempts to force generic `numpy` reduction functions (`numpy.sum`,
`numpy.absolute`) through `dspeed`'s typed `guvectorize`-based processor
framework repeatedly failed for non-obvious reasons (extra positional
arguments like `dtype` that don't fit the framework's type-dispatch
model). The working pattern is: `dspeed` does baseline fitting and
elementwise waveform correction (things it's built for); the actual
charge summation happens in **plain numpy**, after `dspeed` outputs the
already-corrected waveform slice as an array field
(`wf_charge_window`). Don't try to reintroduce a `numpy.sum`-in-`dspeed`
processor — it will very likely hit the same wall again.

## 7. SAr record-length / truncated-decay question — resolved

Earlier single-file inspection suggested the SAr run's (`1450pm_SAr`)
average waveform hadn't fully settled by the end of its 3000-sample
record, raising concern that `total_end: 3000` might systematically
undercount true total light, inflating `psd_param` for every event in
that run (smaller denominator) relative to the liquid run.

This was resolved two independent ways, using the full 42-file production
run (§11):

1. **High-statistics averaging** (870,185 events across 5 files): the
   averaged waveform's tail (last 50 samples) sits at 0.58% of peak
   amplitude, and the tail region (last 200 samples) contributes only
   0.31% of the `charge_total` window's integrated sum. Both negligible —
   the single-file impression of an unsettled tail was mostly visual noise
   that washes out with real statistics.
2. **Corrected time-axis check** (§7a below): at the confirmed 16 ns/sample
   rate, `total_end: 3000` covers 3000 × 16 = 48 μs of real time —
   comfortably longer than argon's triplet lifetime (~1.5 μs, liquid-argon
   reference value). There was never a real risk of the record length
   being too short for the triplet component specifically.

**Conclusion: `total_end: 3000` for this run is not meaningfully biased by
truncation.** This means the SAr run's `psd_param` distribution genuinely
differing from the liquid run's (particularly the 1st percentile — 0.138
vs 0.011) is likely **real physics** — a difference in prompt/total light
fraction between solid and liquid argon phases — not a measurement
artifact. Worth pursuing as an actual analysis result.

## 7a. Waveform sample spacing is 16 ns, not 2 ns — a real correction, and a real DAQ limitation

While investigating §7, cross-referencing `settings.xml`
(`SRV_PARAM_RECLEN`) against the raw waveform data's own `WaveformTable.dt`
field revealed that **every run processed so far (liquid, SAr, and the
plain-numeric-named run) records waveforms at 16 ns/sample, not the 2
ns/sample (500 MS/s) native rate of the DT5730 digitizer.** Confirmed
directly from data (`pc.struct_field(waveform, "dt")`, uniform across
sampled rows in every file checked), not inferred from settings alone.

This is consistent with CoMPASS applying **onboard presumming**: internally
sampling at the native 2 ns rate, then averaging every 8 consecutive raw
samples into 1 before writing to the `.BIN` file (2 ns × 8 = 16 ns). This
is standard CoMPASS/DPP-PSD behavior, not a pipeline bug — but its
consequences hadn't been accounted for anywhere physical-time figures were
derived from sample counts.

**What this affects:**
- **Nothing in the actual pipeline code or dsp config** — every window
  boundary in `dsp_window_configs.yaml` is defined in sample-index terms
  (`baseline_end`, `total_end`, etc.), which remain structurally correct
  regardless of `dt`. Confirmed via full repo grep — no hardcoded
  `sample_ns`/`2.0`/nanosecond assumption exists in `pipeline/`, `config/`,
  or `tests/`.
- **The Gaussian-filter/rise-time systematics investigation** (conducted
  earlier, on the liquid run) used an incorrect `sample_ns=2.0` when
  converting filter strength (σ, in samples) and measured rise times into
  physical nanoseconds. The *relative* pattern found (σ≤3 samples safe,
  σ=5 introduces a measurable bias) is still numerically correct in sample
  units, but every nanosecond figure reported from it was wrong by a factor
  of 8:

  | σ (samples) | Reported (wrong, 2 ns/sample) | **Actual (16 ns/sample)** |
  |---|---|---|
  | 2–3 | 4–6 ns | **32–48 ns** |
  | 5 | 10 ns | **80 ns** |

  Measured 10-90% rise time on the clean, high-amplitude band: reported
  as 56–60 ns, actually **448–480 ns**.

**A more important consequence than the unit correction:** argon's singlet
scintillation component has a lifetime of ~6 ns (liquid-argon reference
value). With a true 16 ns sample spacing, **a single raw sample already
spans nearly 3x the entire singlet decay time** — this holds even with *no*
smoothing applied at all (σ=0). This is not a smoothing-strength tradeoff
to tune; it is a fundamental acquisition-resolution limit. **The singlet
component cannot be time-resolved from this presummed data, under any
offline analysis choice.** The Gaussian-filter question ("how much
smoothing is safe") remains meaningful for the measured ~450 ns rise-time
metric and the microsecond-scale triplet decay, but was never capable of
resolving the singlet in the first place.

**Corrected recommendation** (relative to the rise-time/triplet timescale
this data can actually measure): σ up to 3 samples (48 ns) remains safe —
bias stays near 0% relative to the ~450 ns rise time. σ=5 (80 ns)
introduces the same ~7% relative bias found originally, now correctly
understood as 80 ns against a 448 ns baseline rather than "comparable to
the singlet lifetime."

**Open question for the group, not a pipeline issue:** if resolving the
singlet component directly is important to this project's goals, it
requires a DAQ reconfiguration (disabling or reducing onboard presumming)
for *future* acquisitions — no amount of offline reprocessing of existing
data can recover time resolution that was discarded at acquisition time.

## 8. SPE calibration status

Only the original liquid run (`260520_1340_liquid_1`) has a derived gain
in `config/spe_calibration.yaml` (43,076.6 ± 133.3 ADC·samples/PE,
R²=0.999981, from a 4-peak fit on the self-calibrating low-charge
population — see `pipeline/calibration.py::fit_spe_calibration`). The SAr
run and the plain-numeric-named run both still need their own gain fits.
**Hold off on calibrating the SAr run until §7's truncation question is
resolved** — a calibration derived from biased `charge_total` values would
need to be redone anyway.

## 9. Git/NERSC reconciliation — lessons from a real incident

This repo was built collaboratively across a local machine and NERSC,
which caused significant, repeated confusion, worth documenting so it
isn't repeated:

- **NERSC's local git checkout can silently fall behind** `origin/main`
  if `git pull` isn't run before starting a new editing session there.
  A `git diff` on a stale checkout compares your working file against an
  *old* commit, not against the actual current truth — this produced a
  false alarm at least once (`build_raw_compass.py` appeared to have
  "new" content that was actually already correctly committed).
- **Edits made directly on NERSC via `nano` are invisible to git on any
  other machine until explicitly committed and pushed from NERSC itself.**
  One real fix (`tests/test_calibration.py`'s pytest fixture usage) was
  made this way, never committed, and was nearly lost twice during
  reconciliation — recovered only by cross-referencing the conversation
  history that originally produced it.
- **The practical rule going forward: commit and push immediately after
  any edit, on whichever machine you just edited on — before switching
  machines, before starting a new task.** Do not batch multiple edits
  across a session before syncing. When reconciling drift, diff against
  `origin/main` specifically (`git diff origin/main -- <file>`), not just
  local `HEAD`, and verify with `git log --oneline origin/main` on both
  ends before assuming either side is "ahead."

## 10. Project layout

```
legend-pipeline-parquet/
├── build_raw_compass.py           # parse_run_info(), build_raw_app() — raw stage
├── config/
│   ├── dsp_window_configs.yaml    # per-wf_len dsp window boundaries (§6)
│   └── spe_calibration.yaml       # per-run gain/pedestal (§8)
├── pipeline/
│   ├── __init__.py
│   ├── process.py                 # convert_to_raw, convert_to_dsp,
│   │                               # compute_psd_params, process_file,
│   │                               # select_channel_files, sequence_key
│   ├── calibration.py             # fit_spe_calibration, get_run_key
│   ├── dsp_config.py              # build_dsp_config() — programmatic dspeed chain
│   └── parsl_apps.py              # raw_stage_app, dsp_stage_app, hit_stage_app
├── tests/
│   ├── conftest.py                 # test_daq_file, lh5_reference_hit_path fixtures
│   │                                # (both skip gracefully if data isn't present —
│   │                                #  this repo's tests are expected to partially
│   │                                #  skip on machines without the large test files)
│   ├── test_pipeline_parquet.py    # end-to-end + cross-validation against LH5
│   ├── test_calibration.py         # SPE gain fit regression
│   ├── test_parsl_pipeline.py      # Parsl-vs-direct equivalence (local executors)
│   ├── test_run_parsing.py         # parse_run_info() unit tests, all 4 known conventions
│   └── test_dsp_window_config.py   # build_dsp_config() unit tests
├── utils.py                        # make_local_config() (testing), make_config()
│                                    # (Perlmutter/Slurm, account="m2676")
├── run_pipeline.py                 # CLI batch entry point: --daq-dir, --output-dir, --limit
└── run_parsl_pipeline.py           # Parsl-parallelized batch entry point
```

## 11. Running things

### Local (laptop) — development and fast iteration

```bash
# fast tests always run; slow/data-dependent tests skip gracefully if the
# large test files aren't present on this machine
python -m pytest -v

# sequential (no Parsl), with a limit for cautious first runs against new data
python run_pipeline.py --daq-dir /path/to/some/DAQ/run_folder --output-dir data --limit 2

# Parsl with local ThreadPoolExecutors — exercises the same three-stage
# executor wiring as the real NERSC path, without touching Slurm at all
python test_parsl_local.py   # or equivalent ad hoc script using make_local_config()
```

### NERSC / Perlmutter — real production runs

**One-time setup**, per machine/session:
```bash
cd /global/cfs/cdirs/m2676/users/<you>/legend-pipeline-parquet
git pull origin main              # NERSC's checkout can silently fall behind — see §9
conda activate legend-pipeline-parquet
python -m pytest -v               # confirm the environment itself is healthy first
```

**Running a real batch** — `run_parsl_pipeline.py` submits Slurm jobs on
your behalf via Parsl's `SlurmProvider`; you do not need to be inside an
`salloc`/`srun` session yourself, just run it from a login node:

```bash
python run_parsl_pipeline.py \
  --daq-dir /global/cfs/cdirs/m2676/data/teststands/sarge/sarge9/DAQ/<run_folder> \
  --output-dir /global/cfs/cdirs/m2676/users/<you>/legend-pipeline-parquet/<output_dir> \
  --account m2676 \
  --qos regular \
  [--limit N]
```

**Always stage new data before a full run** — this is not optional caution,
it's how every naming-convention and record-length surprise in this repo's
history was actually caught (§4, §6): `--limit 2` → `--limit 10` → no
limit, checking output at each step (§ "Verifying output" below) before
trusting the next, larger run.

**Watching a submitted job**, in a second terminal:
```bash
squeue -u $USER
```

**After completion, inspect what actually happened** via Slurm accounting
(note: worker-pool jobs show as `CANCELLED` when Parsl tears them down at
the end of a successful run — this is expected, not a failure; see §9-style
note below):
```bash
sacct -u $USER --starttime=today -o JobID,JobName,State,Elapsed,Start,End
```

**If something fails to submit at all**, check `runinfo/<run_number>/` for
Parsl's own logs and the generated Slurm submit script, and check the
account's actual QOS limits directly rather than assuming — limits can be
allocation-specific and differ from NERSC's general documentation
(confirmed the hard way, see "QOS notes" below):
```bash
sacctmgr show assoc user=$USER account=m2676 format=account,qos -p
sacctmgr show qos <qos_name> format=Name,MaxWall -p
```

### QOS notes, learned from real submissions on this account

- **`debug`**: `MaxWall` = 30 minutes on this account. Good for the
  staged `--limit`-based validation runs above; too short for anything
  beyond a handful of files' worth of dsp-stage work.
- **`regular_0` / `regular_1`**: what you actually request is plain
  `qos="regular"` — Slurm internally routes the job into one of these two
  sub-tiers itself; they are not something you choose directly. `MaxWall`
  = 2 days on this account, far more headroom than needed so far.
- A `walltime` of `00:30:00` in `make_config()` has been sufficient for a
  full 42-file run of `run_1450pm_SAr` (~7.27M events, raw+dsp+hit, all
  three Parsl executors) under `regular` QOS. Scale up cautiously for
  substantially larger runs, and check actual elapsed time via `sacct`
  after the fact rather than guessing further in advance.

### Verifying output — do this every time, not just on the first run of a new dataset

```python
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

hit_dir = Path("<output_dir>/hit")
files = sorted(hit_dir.glob("*.parquet"))
print(f"{len(files)} hit-tier files found")
for f in files:
    t = pq.read_table(str(f))
    psd = t.column("psd_param").to_numpy()
    print(f.name, "rows:", t.num_rows, "NaN:", np.isnan(psd).sum())
```

For `run_1450pm_SAr` specifically, a healthy result looks like: every file
at exactly 174,037 rows except the final one (a shorter tail chunk — CoMPASS
appears to split output at a fixed row-count boundary), and NaN counts
(from the noise-floor cut, §-references in the main pipeline doc) clustered
in the teens-to-thirties with no zero or wildly high outliers. Compare any
new run's output against this shape before trusting it.

## 12. What's still open

- **`WANTED_CHANNELS = [1]`** — confirmed as the only channel present
  across all examined runs; still worth confirming with the group that
  channel 1 is definitively the argon-scintillation PMT, not an
  incidental single active channel.
- **The singlet-resolution DAQ limitation (§7a)** — a real, open question
  for the group: does resolving argon's ~6 ns singlet component matter for
  this project's goals? If so, future acquisitions need onboard presumming
  disabled or reduced; no offline fix is possible for already-recorded
  data. Communicate the §7a correction (rise-time/filter figures were off
  by 8x) to whoever received the original, uncorrected numbers.
- **`medium == "unknown"`** for the plain-numeric-named run
  (`run_260521_1023`) — checked `run.info` and `settings.xml` for this run
  specifically (see below); neither records a medium
  (`SW_PARAMETER_CH_LABEL` is the generic default `"CH"`, not a custom
  operator label). This looks like a dead end on the file-metadata side —
  genuinely needs a direct answer from whoever ran this session, not
  further file archaeology.
- **SPE calibration** for the SAr and plain-numeric runs — not yet done.
  SAr is no longer blocked (§7 resolved) and can proceed; the
  plain-numeric run still needs its raw/dsp/hit processing run at all
  (only a `--limit 2` test was planned, not yet executed as of this
  writing).
- **The `.root` (Hcompass.../HcompassF.../HcompassR...) files and
  `run.cae`** found alongside runs' `RAW/`/`FILTERED`/`OFFLINE`
  directories — not investigated in detail. `run.info` and `settings.xml`
  (see below) turned out to be far more useful for acquisition metadata;
  the `.root` files are likely CoMPASS's own histogram/spectrum exports
  and probably don't carry anything this pipeline needs, but this is an
  assumption, not a confirmed fact.
- **The remaining runs under `DAQ/`** (the four other `run_1113am*`
  directories, plus anything else in the tree) have not yet been
  processed through this pipeline — `run_1450pm_SAr` is the only run
  fully validated and run at production scale so far.

### Resolved today, kept here for the record

- **NERSC compute account** — confirmed via Iris and `sacctmgr`: `m2676`
  is a genuine compute allocation, not storage-only.
- **Parsl on real Perlmutter Slurm** — fully exercised: single file →
  small batch (`debug`) → small batch (`regular`, verified numerically
  identical to `debug`) → 10-file batch → full 42-file production run,
  all under `regular` QOS, all verified correct.
- **§7's SAr truncated-decay question** — resolved; see §7 above. Not a
  real bias; `psd_param` differences between SAr and liquid are likely
  genuine physics.
- **§7a's sample-spacing correction** — discovered and quantified;
  affects only previously-reported physical-time figures from the
  Gaussian-filter investigation, not any pipeline code or config.

### Useful acquisition-metadata files discovered today, worth checking for every run going forward

Each run directory (e.g. `DAQ/<run_name>/`) contains, alongside `RAW/`:
- **`run.info`** — plain-text run accounting: start/stop time, real
  duration, per-board readout rate, onboard rejection counts (all zero for
  every run checked so far — no onboard filtering active), input/output
  count rates (real dead time present — `icr`/`ocr` ratio ~10:1 for
  `run_260521_1023`), and the onboard energy-calibration coefficients
  (identity, `c0=0, c1=1, c2=0` — confirms CoMPASS's `energy_calibrated`
  field is genuinely unused, not something the pipeline is missing).
- **`settings.xml`** — the real DAQ configuration: trigger threshold
  (`SRV_PARAM_CH_THRESHOLD`), polarity (`SRV_PARAM_CH_POLARITY`,
  confirmed `POLARITY_NEGATIVE`), pre-trigger (`SRV_PARAM_CH_PRETRG`),
  CoMPASS's own PSD gate widths (`SRV_PARAM_CH_GATE`,
  `SRV_PARAM_CH_GATESHORT`, `SRV_PARAM_CH_GATEPRE`), record length
  (`SRV_PARAM_RECLEN`), channel enable state and label
  (`SRV_PARAM_CH_ENABLED`, `SW_PARAMETER_CH_LABEL`), and whether software
  PSD/energy/time cuts were active (`SW_PARAMETER_CH_*CUTENABLE` — all
  `false` for `run_260521_1023`, i.e. unfiltered data). Parse with
  `xml.etree.ElementTree`, not `grep` — settings are stored as
  `<entry><key>NAME</key><value>...</value></entry>` blocks, not
  directly-named tags.

**Note:** this pipeline's own `charge_prompt`/`charge_total`/`psd_param`
windows were derived independently (by eye, from plotted average
waveforms), not from CoMPASS's own `SRV_PARAM_CH_GATE*` values. Both are
legitimate methodologies, but they are not the same definition of
"prompt"/"total" — worth a deliberate decision (not yet made) about
whether to align the pipeline's windows with CoMPASS's own gate
definition, especially if comparing results against anything computed
from CoMPASS's onboard `energy`/`energy_short` fields directly.
