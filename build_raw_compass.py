import re

#Reads a filename to parse run id, sequence number and medium (if applicable)
def parse_run_info(stem):
    m = re.search(r"_run_(.+)$", stem)
    if m is None:
        raise ValueError(f"cannot parse run from {stem!r}")
    run_tag = m.group(1)

    seq_match = re.search(r"_(\d+)$", run_tag)
    if seq_match:
        seq = int(seq_match.group(1))
        run_base = run_tag[: seq_match.start()]
    else:
        seq = 0
        run_base = run_tag

    medium = "unknown"
    for candidate in ("liquid", "solid", "sar", "lar"):
        if candidate in run_base.lower():
            medium = candidate
            # strip the medium label (and any trailing/leading underscore
            # left behind) from run_base, so run identifies just the
            # date/time/label, with medium tracked separately
            run_base = re.sub(rf"_?{candidate}_?", "_", run_base, flags=re.IGNORECASE).strip("_")
            break

    return run_base, seq, medium


#Streams .BIN through CoMPASS decoder in chunks, writing a compressed parquet
def build_raw_app(inp, out_dir):
    from pathlib import Path

    out = str(Path(out_dir) / "raw" / (Path(inp).stem + ".parquet"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from daq2lh5 import open_stream

    # One worker per core already; keep pyarrow from spawning a thread per core
    # on top of that (nested parallelism would oversubscribe the node).
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    run_base, cycle_id, medium = parse_run_info(Path(inp).stem)

    with open_stream(
        inp,
        in_stream_type="Compass",  # .BIN is auto-detected, but be explicit
        # Each write_table emits >=1 row group, so buffer_size is effectively
        # the Parquet row-group size: ~160 MB at this width (5000-sample wfs).
        buffer_size=16384,
    ) as (streamer, _):
        # All channels of all boards share the one default `CompassEvent`
        # buffer; each row already carries its own `board` and `channel`.
        buffers = streamer.rb_lib["CompassEventDecoder"]
        base = buffers[0].lgdo.view_as("arrow").schema
        base = base.with_metadata(
            {k: v for k, v in base.metadata.items() if k != b"datatype"}
        )
        schema = (
            base.append(pa.field("run", pa.string()))
            .append(pa.field("cycle_id", pa.int64()))
            .append(pa.field("rownumber", pa.int64()))
            .append(pa.field("medium", pa.string()))
        )

        writer = pq.ParquetWriter(
            out,
            schema,
            use_dictionary=False,  # Needed for custom column encoding
            column_encoding={
                "waveform.values.list.element": "DELTA_BINARY_PACKED",
            },
            compression="lz4",
        )

        nrows = 0
        for chunk_list in streamer:
            for rb in chunk_list:
                if rb.out_name != "CompassEvent":
                    continue

                n = rb.loc
                t = rb.lgdo.view_as("arrow").slice(0, n)
                t = (
                    t.append_column("run", pa.array([run_base] * n, type=pa.string()))
                    .append_column("cycle_id", pa.repeat(cycle_id, n))
                    .append_column("rownumber", pa.arange(nrows, nrows + n))
                    .append_column("medium", pa.array([medium] * n, type=pa.string()))
                )
                nrows += n
                writer.write_table(t)

        writer.close()

    return out


if __name__ == "__main__":
    import sys

    print(build_raw_app(sys.argv[1], sys.argv[2]))
