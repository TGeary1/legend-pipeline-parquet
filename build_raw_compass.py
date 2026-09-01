def build_raw_app(inp, out_dir):
    import re
    from pathlib import Path

    out = str(Path(out_dir) / "raw" / (Path(inp).stem + ".parquet"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from daq2lh5 import open_stream

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    m = re.search(r"_run_(\d+)_(\d+)_(\w+)_(\d+)$", Path(inp).stem)
    if m is None:
        raise ValueError(f"cannot parse run from {inp!r}")
    run = int(f"{m.group(1)}{m.group(2)}")
    cycle_id = int(m.group(4))
    medium = m.group(3)  # "liquid" or "solid" — stamped as a string column below

    with open_stream(
        inp, in_stream_type="Compass", buffer_size=16384,
    ) as (streamer, _):
        buffers = streamer.rb_lib["CompassEventDecoder"]
        base = buffers[0].lgdo.view_as("arrow").schema
        base = base.with_metadata(
            {k: v for k, v in base.metadata.items() if k != b"datatype"}
        )
        schema = (
            base.append(pa.field("run", pa.int64()))
            .append(pa.field("cycle_id", pa.int64()))
            .append(pa.field("rownumber", pa.int64()))
            .append(pa.field("medium", pa.string()))
        )

        writer = pq.ParquetWriter(
            out, schema, use_dictionary=False,
            column_encoding={"waveform.values.list.element": "DELTA_BINARY_PACKED"},
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
                    t.append_column("run", pa.repeat(run, n))
                    .append_column("cycle_id", pa.repeat(cycle_id, n))
                    .append_column("rownumber", pa.arange(nrows, nrows + n))
                    .append_column("medium", pa.repeat(medium, n))
                )
                nrows += n
                writer.write_table(t)

        writer.close()

    return out


if __name__ == "__main__":
    import sys
    print(build_raw_app(sys.argv[1], sys.argv[2]))