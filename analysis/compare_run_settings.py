"""
Compare CoMPASS acquisition settings across run folders.

Reads each run's settings.xml (DAQ configuration) and run.info (run
accounting) and prints one table with runs as columns. Rows whose values
differ between runs are marked with '*', so a change in setup between runs
is obvious at a glance.

Why: comparisons between runs (liquid vs solid vs gas) are only meaningful
if the acquisition was configured the same way. Earlier runs in this project
differed in SPE gain and electronics decay time, suggesting different setups.

Standard library only, so it runs anywhere (login node is fine).

Usage (from the repo root):
    python analysis/compare_run_settings.py \\
        --daq-root /global/cfs/cdirs/m2676/data/teststands/sarge/sarge9/DAQ \\
        --runs run_A run_B run_C \\
        --csv results/run_settings.csv
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

SAMPLE_NS = 2.0  # DT5730 native rate; used only to show record length in samples

SETTINGS_KEYS = [
    "SRV_PARAM_RECLEN",          # record length (ns)
    "SRV_PARAM_CH_PRETRG",       # pre-trigger (ns); board default and channel override may both appear
    "SRV_PARAM_CH_THRESHOLD",    # trigger threshold (LSB)
    "SRV_PARAM_CH_POLARITY",
    "SRV_PARAM_CH_GATE",         # CoMPASS long gate (ns)
    "SRV_PARAM_CH_GATESHORT",    # CoMPASS short gate (ns)
    "SRV_PARAM_CH_GATEPRE",
    "SRV_PARAM_CH_ENERGY_COARSE_GAIN",
    "SRV_PARAM_CH_INDYN",        # input dynamic range
    "SRV_PARAM_CH_BLINE_NSMEAN", # onboard baseline averaging
    "SRV_PARAM_CH_TRG_HOLDOFF",
    "SRV_PARAM_CH_DISCR_MODE",
    "SW_PARAMETER_CH_LABEL",
]


def read_settings(path):
    """Return {key: value text}. A key can appear more than once (board-level
    default plus per-channel entries); all occurrences are joined with ','."""
    found = {}
    root = ET.parse(path).getroot()
    for entry in root.iter("entry"):
        key = entry.find("key")
        if key is None or key.text not in SETTINGS_KEYS:
            continue
        val = entry.find("value")
        if val is None:
            continue
        inner = val.find("value")
        text = (inner if inner is not None else val).text
        found.setdefault(key.text, []).append(str(text))
    return {k: ",".join(v) for k, v in found.items()}


def read_run_info(path):
    """Pull start time, duration, and input/output trigger rates from run.info."""
    raw = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            raw[k.strip()] = v.strip()
    info = {"time.start": raw.get("time.start", ""), "time.real": raw.get("time.real", "")}
    icr = [v for k, v in raw.items() if k.endswith(".icr")]
    ocr = [v for k, v in raw.items() if k.endswith(".ocr")]
    info["icr (in/s)"] = ",".join(icr)
    info["ocr (rec/s)"] = ",".join(ocr)
    try:
        info["recorded fraction"] = f"{float(ocr[0]) / float(icr[0]):.3f}"
    except (IndexError, ValueError, ZeroDivisionError):
        info["recorded fraction"] = ""
    return info


def summarize_run(run_dir):
    row = {}
    settings_path = run_dir / "settings.xml"
    info_path = run_dir / "run.info"
    if settings_path.exists():
        row.update(read_settings(settings_path))
        try:
            reclen = float(row["SRV_PARAM_RECLEN"].split(",")[0])
            row["samples @ 2 ns"] = str(int(round(reclen / SAMPLE_NS)))
        except (KeyError, ValueError):
            pass
    else:
        row["NOTE"] = "no settings.xml"
    if info_path.exists():
        row.update(read_run_info(info_path))
    bins = sorted((run_dir / "RAW").glob("*.BIN")) if (run_dir / "RAW").exists() else []
    row["BIN files"] = str(len(bins))
    row["BIN size (GB)"] = f"{sum(b.stat().st_size for b in bins) / 1e9:.1f}"
    return row


def main(args):
    daq_root = Path(args.daq_root)
    run_names = args.runs or sorted(p.name for p in daq_root.glob(args.glob) if p.is_dir())
    runs = {name: summarize_run(daq_root / name) for name in run_names}

    rows = []
    for r in runs.values():
        for k in r:
            if k not in rows:
                rows.append(k)

    width = max(len(r) for r in rows) + 2
    col = max(14, max(len(n) for n in run_names) + 2)
    print(" " * (width + 2) + "".join(n.ljust(col) for n in run_names))
    for key in rows:
        values = [runs[n].get(key, "-") for n in run_names]
        differs = len(set(values)) > 1
        mark = "* " if differs else "  "
        print(mark + key.ljust(width) + "".join(v[:col - 2].ljust(col) for v in values))
    print("\n'*' = value differs between runs")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["setting"] + run_names)
            for key in rows:
                w.writerow([key] + [runs[n].get(key, "") for n in run_names])
        print(f"written to {args.csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daq-root", required=True, help="directory containing the run folders")
    p.add_argument("--runs", nargs="*", help="run folder names to compare (default: all matching --glob)")
    p.add_argument("--glob", default="run_*", help="pattern for run folders when --runs is not given")
    p.add_argument("--csv", default=None, help="optional path to also save the table as CSV")
    main(p.parse_args())
