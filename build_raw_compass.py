import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa

log = logging.getLogger(__name__)

# DT5730 native rate. Used only as a fallback when a run has no settings.xml.
# daq2lh5's CoMPASS decoder stores dt = 16 as a hardcoded default; it is wrong
# for this digitizer and is overwritten below (see PIPELINE_OVERVIEW_PARQUET.md §7a).
DEFAULT_SAMPLE_NS = 2.0


def parse_run_info(stem):
    """Extract (run_base, sequence, medium) from a CoMPASS filename stem.

    Everything after '_run_' is the run tag. A leading YYMMDD_HHMM date-time is
    part of the run identity, so it is split off before looking for a trailing
    sequence number; otherwise the unnumbered first file of a date-time run
    (e.g. run_260520_1447) would be misread as sequence 1447. A trailing bare
    integer after that is the sequence number; none means sequence 0 (the
    first file of the run). A known medium keyword is stripped from run_base
    and reported separately.
    """
    m = re.search(r"_run_(.+)$", stem)
    if m is None:
        raise ValueError(f"cannot parse run from {stem!r}")
    run_tag = m.group(1)

    m_dt = re.match(r"(\d{6}_\d{4})(.*)$", run_tag)
    head, rest = (m_dt.group(1), m_dt.group(2)) if m_dt else ("", run_tag)

    seq_match = re.search(r"_(\d+)$", rest)
    if seq_match:
        seq = int(seq_match.group(1))
        rest = rest[: seq_match.start()]
    else:
        seq = 0
    run_base = (head + rest).strip("_")

    medium = "unknown"
    for candidate in ("liquid", "solid", "gas", "sar", "lar", "gar"):
        if candidate in run_base.lower():
            medium = candidate
            run_base = re.sub(rf"_?{candidate}_?", "_", run_base, flags=re.IGNORECASE).strip("_")
            break

    return run_base, seq, medium


def read_reclen_ns(daq_path):
    """Record length in ns from the run's settings.xml.
    Files live in <run>/RAW/, so settings.xml is two levels up."""
    settings = Path(daq_path).parent.parent / "settings.xml"
    if not settings.exists():
        return None
    for entry in ET.parse(settings).getroot().iter("entry"):
        key = entry.find("key")
        if key is not None and key.text == "SRV_PARAM_RECLEN":
            val = entry.find("value")
            inner = val.find("value") if val is not None else None
            try:
                return float((inner if inner is not None else val).text)
            except (TypeError, ValueError):
                return None
    return None


def sample_period_ns(daq_path, n_samples):
    """Sample spacing = record length / samples per waveform."""
    reclen = read_reclen_ns(daq_path)
    if reclen is None:
        log.warning(f"no settings.xml record length for {daq_path}; assuming {DEFAULT_SAMPLE_NS} ns/sample")
        return DEFAULT_SAMPLE_NS
    dt = reclen / n_samples
    if not np.isclose(dt, DEFAULT_SAMPLE_NS):
        log.warning(f"{daq_path}: sample spacing {dt} ns differs from the expected {DEFAULT_SAMPLE_NS} ns")
    return dt


def set_waveform_dt(table, dt_ns):
    """Replace the 'dt' field inside the waveform struct column."""
    idx = table.schema.get_field_index("waveform")
    col = table.column(idx)
    new_chunks = []
    for chunk in col.chunks:
        fields = list(chunk.type)
        children = chunk.flatten()          # respects slice offsets, unlike .field()
        children = [pa.array(np.full(len(chunk), dt_ns), type=f.type) if f.name == "dt" else c
                    for f, c in zip(fields, children)]
        new_chunks.append(pa.StructArray.from_arrays(children, fields=fields))
    return table.set_column(idx, table.schema.field(idx), pa.chunked_array(new_chunks, type=col.type))


def build_raw_app(inp, out_dir):
    import pyarrow.parquet as pq
    from daq2lh5 import open_stream

    out = str(Path(out_dir) / "raw" / (Path(inp).stem + ".parquet"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = out + ".partial"   # written here, renamed to `out` only once complete

    # One worker per core already; keep pyarrow from spawning a thread per core
    # on top of that (nested parallelism would oversubscribe the node).
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    run_base, cycle_id, medium = parse_run_info(Path(inp).stem)

    try:
        with open_stream(
            inp,
            in_stream_type="Compass",  # .BIN is auto-detected, but be explicit
            # Each write_table emits >=1 row group, so buffer_size is effectively
            # the Parquet row-group size.
            buffer_size=16384,
        ) as (streamer, _):
            # All channels of all boards share the one default `CompassEvent`
            # buffer; each row already carries its own `board` and `channel`.
            buffers = streamer.rb_lib["CompassEventDecoder"]
            base = buffers[0].lgdo.view_as("arrow").schema
            base = base.with_metadata(
                {k: v for k, v in base.metadata.items() if k != b"datatype"}
            )
            n_samples = base.field("waveform").type.field("values").type.list_size
            dt_ns = sample_period_ns(inp, n_samples)

            schema = (
                base.append(pa.field("run", pa.string()))
                .append(pa.field("cycle_id", pa.int64()))
                .append(pa.field("rownumber", pa.int64()))
                .append(pa.field("medium", pa.string()))
            )

            writer = pq.ParquetWriter(
                tmp,
                schema,
                use_dictionary=False,  # Needed for custom column encoding
                column_encoding={"waveform.values.list.element": "DELTA_BINARY_PACKED"},
                compression="lz4",
            )
            try:
                nrows = 0
                for chunk_list in streamer:
                    for rb in chunk_list:
                        if rb.out_name != "CompassEvent":
                            continue
                        n = rb.loc
                        t = rb.lgdo.view_as("arrow").slice(0, n)
                        t = set_waveform_dt(t, dt_ns)
                        t = (
                            t.append_column("run", pa.array([run_base] * n, type=pa.string()))
                            .append_column("cycle_id", pa.repeat(cycle_id, n))
                            .append_column("rownumber", pa.arange(nrows, nrows + n))
                            .append_column("medium", pa.array([medium] * n, type=pa.string()))
                        )
                        nrows += n
                        writer.write_table(t)
            finally:
                writer.close()

        os.replace(tmp, out)   # atomic: `out` only ever appears complete
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return out


if __name__ == "__main__":
    import sys

    print(build_raw_app(sys.argv[1], sys.argv[2]))
