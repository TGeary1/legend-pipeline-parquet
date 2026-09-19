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

## 7. An open, unresolved physics question — flag before using SAr data for anything comparative

The `1450pm_SAr` run's average waveform does **not** fully return to
baseline by the end of its 3000-sample record (still visibly recovering
at sample 3000/3000), unlike the liquid run's clean full recovery by
~sample 2000/5000. This means `total_end: 3000` for that run almost
certainly **undercounts true total light**, which would systematically
inflate `psd_param` for every event in that run (smaller denominator).

Consistent with this: the SAr run's `psd_param` distribution's 1st
percentile (0.138) is over 10x higher than the liquid run's (0.011),
while the medians are closer (0.408 vs 0.341). This is exactly the pattern
you'd expect from `total_end` being genuinely too short, not necessarily
a difference in real singlet/triplet physics between the two media.

**Do not draw any liquid-vs-solid comparison from current `psd_param` or
`n_pe` values without first resolving whether the record length was a
deliberate, sufficient acquisition choice for solid argon, or a limitation
that needs a longer record in future runs.** This is a real, open question
for the group — not a pipeline bug to silently work around.

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

```bash
# local testing (fast tests always run; slow/data-dependent tests skip
# gracefully if the large test files aren't present)
python -m pytest -v

# a real batch, with a limit for cautious first runs against new data
python run_pipeline.py --daq-dir /path/to/some/DAQ/run_folder --output-dir data --limit 2
```

On NERSC, activate the matching conda environment first
(`conda activate legend-pipeline-parquet`) — see `utils.py` for how this
gets wired into the Parsl `SlurmProvider`'s `worker_init` for actual
Perlmutter submission.

## 12. What's still open

- **NERSC compute account/QOS** — `m2676` is the working assumption
  (matches the storage paths used throughout), pending confirmation via
  Iris that it's also a valid *compute* association, not just storage.
- **`WANTED_CHANNELS = [1]`** — confirmed as the only channel present
  across all examined runs; still worth confirming with the group that
  channel 1 is definitively the argon-scintillation PMT, not an
  incidental single active channel.
- **§7's SAr truncated-decay question** — unresolved, blocks any
  liquid-vs-solid comparison.
- **`medium == "unknown"`** for the plain-numeric-named runs — needs
  clarification from the group on what medium those runs used.
- **SPE calibration** for the SAr and plain-numeric runs — not yet done.
- **The `.root` (Hcompass...) file** found alongside some runs' `RAW/`
  directories — not investigated; likely a CoMPASS histogram/summary
  export in ROOT format, unrelated to this pipeline's per-event waveform
  processing, but worth a low-priority confirmation with whoever manages
  the acquisition.
- **Parsl on real Perlmutter Slurm** — local-executor equivalence is
  verified; the actual `SlurmProvider` path has not yet been exercised
  against a real allocation.
