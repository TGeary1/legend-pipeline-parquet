"""
Slow (triplet) lifetime of the alpha light, measured in blocks of files
across each run, or in chosen time windows.

Each block gets its own forward fit (the model and fit from
forward_fit_analysis.py): the average clean bright alpha waveform is fitted
with  prompt + intermediate + slow exponential + floor, convolved with the
run's measured single-photoelectron response.

What this tests: when the slow light changes (the gas run's swing, the
solid's evening-to-morning change), does the slow LIFETIME change with it?
  - lifetime shorter when slow light drops -> impurities quenching the
    triplet state (or a temperature effect on it)
  - lifetime unchanged -> fewer triplet states made, or light lost, not
    quenching

Event selection, per block:
  - n_pe >= --n-pe-min (bright: the time profile is only visible in bright
    events; keep it equal across compared blocks)
  - not saturated (lowest_adc above the clipping floor + margin)
  - psd_param within --psd-half-width of the block's own alpha-band peak
    (keeps one population; 0 disables). Note this selects on the very shape
    being fitted, so keep the band wide.
  - max_jump_pe below --max-jump (pileup; from the hit tier). Fits are also
    repeated at the --check-jumps thresholds; the spread is a systematic.

Memory: waveforms are read in batches and only running sums are kept, so a
process needs ~1-2 GB whatever the block size. Files are processed in
parallel (--workers). Run on a compute node:
    salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m2676

Usage (from the repo root):
    python analysis/lifetime_vs_time.py --workers 32 --output-dir results/lifetime
    python analysis/lifetime_vs_time.py --workers 32 --output-dir results/lifetime_windows \\
        --window liquid_early:260520_1340:0-8 --window solid_1800:260520_1800:70-172
"""
import argparse
import json
import pickle
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from build_raw_compass import parse_run_info  # noqa: E402
from forward_fit_analysis import SAMPLE_NS, clean_response, make_model, run_fit  # noqa: E402

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PHASE_COLOR = {"gas": "#2a78d6", "liquid": "#eb6834", "solid": "#1baf7a"}
HIT_COLS = ["n_pe", "psd_param", "lowest_adc", "max_jump_pe"]


def style(ax):
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Finding runs and grouping their files
# ---------------------------------------------------------------------------

def find_runs(args):
    """Run folders that have hit and dsp tiers and a phase in run_phases.yaml,
    each with its files in acquisition order."""
    with open(args.phases) as f:
        phase_db = {str(k).replace("_", ""): v for k, v in (yaml.safe_load(f) or {}).items()}
    runs = {}
    for rd in sorted(Path(args.data_root).glob(args.run_glob)):
        if not (rd / "hit").exists() or not (rd / "dsp").exists():
            continue
        files = sorted((rd / "hit").glob("*.parquet"), key=lambda f: parse_run_info(f.stem)[1])
        if not files:
            continue
        run_base = parse_run_info(files[0].stem)[0]
        phase = phase_db.get(run_base.replace("_", ""))
        if phase is None or run_base in runs:
            continue
        runs[run_base] = {"dir": rd, "phase": phase,
                          "files": [(parse_run_info(f.stem)[1], f.name) for f in files]}
    return runs


def make_groups(runs, args):
    """The sets of files to fit: --window entries if given, else consecutive
    blocks of --block-files files in every run."""
    groups = []
    if args.window:
        for w in args.window:
            label, run_base, file_range = w.split(":")
            first, last = (int(x) for x in file_range.split("-"))
            if run_base not in runs:
                log.warning(f"window {label}: run {run_base} not found; skipped")
                continue
            names = [n for s, n in runs[run_base]["files"] if first <= s <= last]
            groups.append({"label": label, "run": run_base, "files": names,
                           "first": first, "last": last})
    else:
        for run_base, r in runs.items():
            files = r["files"]
            for i in range(0, len(files), args.block_files):
                chunk = files[i:i + args.block_files]
                groups.append({"label": f"{run_base}:{chunk[0][0]}-{chunk[-1][0]}", "run": run_base,
                               "files": [n for _, n in chunk],
                               "first": chunk[0][0], "last": chunk[-1][0]})
    return groups


# ---------------------------------------------------------------------------
# Pass 1 (hit tier only): each group's alpha band
# ---------------------------------------------------------------------------

def group_band(run_dir, names, args):
    """PSD peak of the group's bright, unsaturated events, +/- half-width."""
    if args.psd_half_width <= 0:
        return None
    psd_all = []
    for name in names:
        t = pq.read_table(str(Path(run_dir) / "hit" / name), columns=HIT_COLS)
        n_pe, psd, low = (t.column(c).to_numpy() for c in HIT_COLS[:3])
        sel = (n_pe >= args.n_pe_min) & (low >= args.sat_floor + args.sat_margin) & np.isfinite(psd)
        psd_all.append(psd[sel])
    psd_all = np.concatenate(psd_all)
    edges = np.linspace(-0.1, 1.1, 121)
    counts, _ = np.histogram(psd_all, bins=edges)
    c = 0.5 * (edges[:-1] + edges[1:])[np.argmax(np.convolve(counts, np.ones(3) / 3, mode="same"))]
    return float(c - args.psd_half_width), float(c + args.psd_half_width)


# ---------------------------------------------------------------------------
# Pass 2 (dsp waveforms): running sums per file, run in parallel
# ---------------------------------------------------------------------------

def file_sums(job):
    """One file: sum of 1 PE waveforms, and for each pileup threshold the
    sum, sum of squares and count of clean bright alpha waveforms."""
    dsp_path, hit_path, band, cfg = job
    t = pq.read_table(hit_path, columns=HIT_COLS)
    n_pe, psd, low, jump = (t.column(c).to_numpy() for c in HIT_COLS)

    is_spe = (n_pe > 0.85) & (n_pe < 1.15)
    bright = ((n_pe >= cfg["n_pe_min"]) & (low >= cfg["sat_floor"] + cfg["sat_margin"])
              & np.isfinite(psd))
    if band is not None:
        bright &= (psd >= band[0]) & (psd < band[1])
    thresholds = cfg["thresholds"]

    out = {"spe_sum": None, "n_spe": 0,
           "sum": {thr: None for thr in thresholds},
           "sumsq": {thr: None for thr in thresholds},
           "n": {thr: 0 for thr in thresholds}}

    def add(store, key, values):
        s = values.sum(axis=0, dtype=np.float64)
        store[key] = s if store[key] is None else store[key] + s

    offset = 0
    pf = pq.ParquetFile(dsp_path)
    for batch in pf.iter_batches(batch_size=cfg["batch_size"], columns=["wf_charge_window"]):
        vals = pc.struct_field(batch.column("wf_charge_window"), "values")
        wfs = np.stack(vals.to_numpy(zero_copy_only=False)).astype(np.float32)
        rows = slice(offset, offset + len(wfs))
        offset += len(wfs)

        spe = wfs[is_spe[rows]]
        if len(spe):
            add(out, "spe_sum", spe)
            out["n_spe"] += len(spe)
        b, j = bright[rows], jump[rows]
        for thr in thresholds:
            w = wfs[b & (j < thr)]
            if len(w):
                add(out["sum"], thr, w)
                add(out["sumsq"], thr, w.astype(np.float64) ** 2)
                out["n"][thr] += len(w)
    if offset != len(n_pe):
        raise ValueError(f"row mismatch: {dsp_path} has {offset} rows, hit has {len(n_pe)}")
    return out


def combine(parts):
    """Add up the per-file sums of one group (or one run)."""
    def add(a, b):
        return b if a is None else (a if b is None else a + b)
    tot = {"spe_sum": None, "n_spe": 0, "sum": {}, "sumsq": {}, "n": {}}
    for p in parts:
        tot["spe_sum"] = add(tot["spe_sum"], p["spe_sum"])
        tot["n_spe"] += p["n_spe"]
        for thr in p["n"]:
            tot["sum"][thr] = add(tot["sum"].get(thr), p["sum"][thr])
            tot["sumsq"][thr] = add(tot["sumsq"].get(thr), p["sumsq"][thr])
            tot["n"][thr] = tot["n"].get(thr, 0) + p["n"][thr]
    return tot


def mean_and_error(tot, thr):
    n = tot["n"][thr]
    if n < 2:
        return None, None, n
    mean = tot["sum"][thr] / n
    var = np.maximum(tot["sumsq"][thr] / n - mean ** 2, 0)
    return mean, np.sqrt(var / n), n


# ---------------------------------------------------------------------------
# Fitting and output
# ---------------------------------------------------------------------------

TAU_STARTS_NS = (1500, 3000, 5000)


def best_fit(M, M_err, model, sys_frac, starts=TAU_STARTS_NS):
    """Fit from each starting lifetime in 'starts' and keep the lowest chi2, so a fit
    that gets stuck at its starting guess can't be reported as the answer."""
    best = None
    for start in starts:
        try:
            r = run_fit(M, M_err, model, sys_frac=sys_frac, tau_slow_start_ns=start)
        except (RuntimeError, ValueError):
            continue
        if best is None or r["chi2_red"] < best["chi2_red"]:
            best = r
    return best


def fit_group(tot, model, args):
    """Baseline fit at --max-jump, plus the --check-jumps fits; the spread of
    tau_slow across them is the pileup-cut systematic."""
    M, M_err, n = mean_and_error(tot, args.max_jump)
    if M is None or n < args.min_events:
        return None
    base = best_fit(M, M_err, model, args.sys_frac)
    if base is None:
        log.warning("fit failed from every starting lifetime")
        return None
    taus = [base["tau_slow_ns"]]
    checks = {}
    for thr in args.check_jumps:
        Mx, Mx_err, nx = mean_and_error(tot, thr)
        if Mx is None:
            continue
        r = best_fit(Mx, Mx_err, model, args.sys_frac, starts=(base["tau_slow_ns"],))
        if r is None:
            continue
        checks[f"jump<{thr:g}"] = {"tau_slow_ns": r["tau_slow_ns"], "slow_frac": r["slow_frac"], "n": nx}
        taus.append(r["tau_slow_ns"])
    return {
        "n_events": n,
        "tau_slow_ns": base["tau_slow_ns"],
        "tau_slow_stat_err_ns": base["tau_slow_stat_err_ns"],
        "tau_slow_range_ns": [min(taus), max(taus)],
        "slow_frac": base["slow_frac"],
        "tau_inter_ns": base["tau_inter_ns"],
        "floor_share": base["floor_share"],
        "chi2_red": base["chi2_red"],
        "params_at_bound": base["params_at_bound"],
        "checks": checks,
        "M": M, "curve": base["curve"],
    }

def fit_one(job):
    """All fits for one group, run in a worker process. The model is built
    here because a function made inside another function (a closure) can't
    be sent between processes; the plain arrays can."""
    label, tot, R, args = job
    sums = [v for v in tot["sum"].values() if v is not None]
    if not sums:
        return label, None
    fit = fit_group(tot, make_model(R, len(sums[0])), args)
    log.info(f"fitted {label}")
    return label, fit


def plot_time_series(groups, runs, path):
    """tau_slow and slow fraction per group, on the same file-number axis as
    stability_vs_time.py (runs side by side, in name order)."""
    offsets, x0 = {}, 0
    for run_base, r in runs.items():
        offsets[run_base] = x0
        x0 += len(r["files"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for g in groups:
        f = g.get("fit")
        if f is None:
            continue
        idx = [i for i, (_, n) in enumerate(runs[g["run"]]["files"]) if n in set(g["files"])]
        x = offsets[g["run"]] + 0.5 * (idx[0] + idx[-1])
        color = PHASE_COLOR.get(runs[g["run"]]["phase"], "#777777")
        lo, hi = f["tau_slow_range_ns"]
        ax1.errorbar(x, f["tau_slow_ns"] / 1000, yerr=[[(f["tau_slow_ns"] - lo) / 1000], [(hi - f["tau_slow_ns"]) / 1000]],
                     fmt="o", color=color, markersize=5, capsize=2, linewidth=1)
        ax2.plot(x, f["slow_frac"], "o", color=color, markersize=5)
    for run_base, x in offsets.items():
        for ax in (ax1, ax2):
            ax.axvline(x - 0.5, color="#999999", linewidth=0.8)
        ax1.text(x + len(runs[run_base]["files"]) / 2, 1.02, f"{run_base}\n{runs[run_base]['phase']}",
                 transform=ax1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("slow lifetime τ (μs)\nbars: pileup-cut range")
    ax2.set_ylabel("slow fraction of light")
    ax2.set_xlabel("file number across the dataset (each file ≈ 40 s; runs in name order)")
    for ax in (ax1, ax2):
        style(ax)
    for phase, color in PHASE_COLOR.items():
        ax1.plot([], [], "o", color=color, label=phase)
    ax1.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_group_fit(g, path):
    """Average waveform and fitted model for one group, log scale."""
    f = g["fit"]
    t_us = np.arange(len(f["M"])) * SAMPLE_NS / 1000
    peak = f["M"].max()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(t_us, np.clip(f["M"] / peak, 1e-4, None), color="#2a78d6", linewidth=1, label="data")
    ax.semilogy(t_us, np.clip(f["curve"] / peak, 1e-4, None), "--", color="#eb6834", linewidth=1.5,
                label="model ⊗ 1 PE response")
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("time from start of charge window (μs)")
    ax.set_ylabel("normalized amplitude")
    ax.set_title(f"{g['label']}: τ = {f['tau_slow_ns']:.0f} ns, slow fraction {f['slow_frac']:.3f}, "
                 f"χ²/ndf {f['chi2_red']:.2f}")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args)
    if args.runs:
        runs = {k: v for k, v in runs.items() if k in args.runs}
    groups = make_groups(runs, args)
    log.info(f"{len(runs)} runs, {len(groups)} groups to fit")

    cfg = {"n_pe_min": args.n_pe_min, "sat_floor": args.sat_floor, "sat_margin": args.sat_margin,
           "thresholds": sorted({args.max_jump, *args.check_jumps}), "batch_size": args.batch_size}

    # pass 1: each group's alpha band (hit tier only, cheap)
    for g in groups:
        g["band"] = group_band(runs[g["run"]]["dir"], g["files"], args)

    # pass 2: per-file sums, all files of all groups, in parallel
    jobs, keys = [], []
    for gi, g in enumerate(groups):
        rd = runs[g["run"]]["dir"]
        for name in g["files"]:
            jobs.append((str(rd / "dsp" / name), str(rd / "hit" / name), g["band"], cfg))
            keys.append(gi)
    log.info(f"reading {len(jobs)} files with {args.workers} workers")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            parts = list(pool.map(file_sums, jobs))
    else:
        parts = [file_sums(j) for j in jobs]

    # Everything that changes the sums. A saved sums.pkl is reused only if
    # this matches exactly; otherwise the files are read again.
    key = {"cfg": cfg, "psd_half_width": args.psd_half_width,
           "groups": [(g["run"], g["files"]) for g in groups]}
    cache = out / "sums.pkl"
    saved = None
    if cache.exists():
        with open(cache, "rb") as fh:
            saved = pickle.load(fh)
        if saved["key"] != key:
            log.info("saved sums were made with different settings; re-reading the files")
            saved = None

    if saved is not None:
        log.info(f"reusing saved sums from {cache}")
        totals, responses = saved["totals"], saved["responses"]
        for g, band in zip(groups, saved["bands"]):
            g["band"] = band
    else:
        # pass 1: each group's alpha band (hit tier only, cheap)
        for g in groups:
            g["band"] = group_band(runs[g["run"]]["dir"], g["files"], args)

        # pass 2: per-file sums, all files of all groups, in parallel
        jobs, keys = [], []
        for gi, g in enumerate(groups):
            rd = runs[g["run"]]["dir"]
            for name in g["files"]:
                jobs.append((str(rd / "dsp" / name), str(rd / "hit" / name), g["band"], cfg))
                keys.append(gi)
        log.info(f"reading {len(jobs)} files with {args.workers} workers")
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                parts = list(pool.map(file_sums, jobs))
        else:
            parts = [file_sums(j) for j in jobs]

        # 1 PE response per run, from every file read for that run
        responses = {}
        for run_base in {g["run"] for g in groups}:
            tot = combine([p for p, gi in zip(parts, keys) if groups[gi]["run"] == run_base])
            R, tau_elec = clean_response(tot["spe_sum"] / max(tot["n_spe"], 1))
            responses[run_base] = (R, tau_elec)
            log.info(f"{run_base}: 1 PE response from {tot['n_spe']:,} events, electronics τ {tau_elec:.0f} ns")
        totals = [combine([p for p, k in zip(parts, keys) if k == gi]) for gi in range(len(groups))]

        with open(cache, "wb") as fh:
            pickle.dump({"key": key, "totals": totals, "responses": responses,
                         "bands": [g["band"] for g in groups]}, fh)
        log.info(f"sums saved to {cache}")

    # fits: one group per worker process
    jobs = [(g["label"], tot, responses[g["run"]][0], args) for g, tot in zip(groups, totals)]
    log.info(f"fitting {len(jobs)} groups with {args.workers} workers")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fits = dict(pool.map(fit_one, jobs))
    else:
        fits = dict(fit_one(j) for j in jobs)
    for g in groups:
        g["fit"] = fits[g["label"]]
        if g["fit"] is not None:
            plot_group_fit(g, out / f"fit_{g['label'].replace(':', '_')}.png")

    if not args.window:
        plot_time_series(groups, runs, out / "lifetime_vs_time.png")

    print(f"\n{'group':<26}{'phase':<8}{'events':>9}{'τ_slow (ns)':>16}{'range (ns)':>14}"
          f"{'slow frac':>11}{'τ_int (ns)':>11}{'floor':>7}{'χ²/ndf':>9}")
    for g in groups:
        f = g["fit"]
        phase = runs[g["run"]]["phase"]
        if f is None:
            print(f"{g['label']:<26}{phase:<8}   fit failed or too few events")
            continue
        lo, hi = f["tau_slow_range_ns"]
        flag = "  (bound: " + ", ".join(f["params_at_bound"]) + ")" if f["params_at_bound"] else ""
        print(f"{g['label']:<26}{phase:<8}{f['n_events']:>9,}{f['tau_slow_ns']:>9.0f} ± {f['tau_slow_stat_err_ns']:<4.0f}"
              f"{lo:>7.0f}-{hi:<6.0f}{f['slow_frac']:>11.3f}{f['tau_inter_ns']:>11.0f}{f['floor_share']:>7.3f}"
              f"{f['chi2_red']:>9.2f}{flag}")

    strip = lambda f: None if f is None else {k: v for k, v in f.items() if k not in ("M", "curve")}
    summary = {"settings": {k: v for k, v in vars(args).items()},
               "electronics_tau_ns": {k: v[1] for k, v in responses.items()},
               "groups": [{**{k: v for k, v in g.items() if k != "fit"}, "fit": strip(g["fit"])}
                          for g in groups]}
    with open(out / "lifetime_vs_time.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log.info(f"done: results in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-glob", default="data_*")
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--phases", default=str(REPO_ROOT / "config/run_phases.yaml"))
    ap.add_argument("--runs", nargs="*", help="Only these run_base values, e.g. 260520_1239")
    ap.add_argument("--output-dir", default="results/lifetime")
    ap.add_argument("--window", action="append",
                    help="LABEL:RUN:FIRST-LAST (file numbers, inclusive); repeatable. Replaces blocks.")
    ap.add_argument("--block-files", type=int, default=10, help="Files per block when no --window is given")
    ap.add_argument("--n-pe-min", type=float, default=30, help="Bright threshold; keep equal across compared groups")
    ap.add_argument("--psd-half-width", type=float, default=0.15, help="Alpha band half-width in PSD; 0 disables")
    ap.add_argument("--sat-floor", type=float, default=7067)
    ap.add_argument("--sat-margin", type=float, default=50)
    ap.add_argument("--max-jump", type=float, default=8.0, help="Pileup cut for the baseline fit, in PE")
    ap.add_argument("--check-jumps", type=float, nargs="*", default=[5.0, 12.0],
                    help="Extra pileup thresholds; their spread is a systematic")
    ap.add_argument("--sys-frac", type=float, default=0.01, help="Per-sample systematic, fraction of the average")
    ap.add_argument("--min-events", type=int, default=2000, help="Skip groups with fewer clean events")
    ap.add_argument("--workers", type=int, default=1, help="Parallel processes (~1-2 GB each)")
    ap.add_argument("--batch-size", type=int, default=5000, help="Waveforms read at a time per process")
    main(ap.parse_args())
