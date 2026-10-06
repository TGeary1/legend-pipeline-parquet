def make_local_config(max_threads=4):
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor

    return Config(executors=[
        ThreadPoolExecutor(max_threads=max_threads, label="raw"),
        ThreadPoolExecutor(max_threads=max_threads, label="dsp"),
        ThreadPoolExecutor(max_threads=max_threads, label="hit"),
    ])


# Per-job time limits on this account (sacctmgr): debug = 30 min, regular = 2 days.
# Requesting more than a QOS allows is rejected at submission (QOSMaxWallDurationPerJobLimit).
DEFAULT_WALLTIME = {"debug": "00:30:00", "regular": "02:00:00"}

# Workers per node for each stage, set by memory, not by core count.
# A Perlmutter CPU node has 512 GB. Measured peak memory for one dsp task
# on a 5000-sample file is ~10 GB (it holds the whole waveform table in
# memory), so 32 dsp workers use ~330 GB and leave headroom. Without a cap,
# Parsl starts one worker per 2 cores (128), which ran the nodes out of
# memory and killed the overnight solid runs (WorkerLost / ManagerLost).
# raw is lighter (streams the BIN file), so it can run more at once.
DEFAULT_WORKERS_PER_NODE = {"raw": 64, "dsp": 32, "hit": 32}


def make_config(nodes=1, account="m2676", qos="regular", walltime=None, workers_per_node=None):
    from parsl.config import Config
    from parsl.providers import SlurmProvider
    from parsl.launchers import SrunLauncher
    from parsl.executors import HighThroughputExecutor

    if walltime is None:
        walltime = DEFAULT_WALLTIME.get(qos, "00:30:00")
    if workers_per_node is None:
        workers_per_node = DEFAULT_WORKERS_PER_NODE

    # Compute-node workers don't inherit the login shell's directory or
    # PYTHONPATH; without this, `import pipeline` fails on the workers.
    repo_dir = "/global/cfs/cdirs/m2676/users/tgeary/legend-pipeline-parquet"
    worker_init = (
        f"cd {repo_dir} && "
        "module load conda && "
        "conda activate legend-pipeline-parquet && "
        f"export PYTHONPATH={repo_dir}:$PYTHONPATH"
    )

    def make_executor(label):
        return HighThroughputExecutor(
            label=label,
            cores_per_worker=2,
            max_workers_per_node=workers_per_node[label],
            provider=SlurmProvider(
                # keyword, not positional: SlurmProvider's first positional
                # argument is `partition`, so passing qos positionally left
                # the QOS at its default (debug, 30 min cap).
                qos=qos,
                account=account,
                nodes_per_block=nodes,
                scheduler_options="#SBATCH -C cpu",
                worker_init=worker_init,
                launcher=SrunLauncher(overrides="-c 128"),
                walltime=walltime,
                cmd_timeout=120,
            ),
        )

    return Config(executors=[
        make_executor("raw"),
        make_executor("dsp"),
        make_executor("hit"),
    ])
