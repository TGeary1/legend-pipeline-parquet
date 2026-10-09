"""Parsl configurations: local threads for testing, Slurm on NERSC Perlmutter
for production. Each pipeline stage (raw, dsp, hit) gets its own executor so
its worker count can be set by that stage's memory use."""
from pipeline.paths import REPO_ROOT

# Per-job time limits on NERSC (sacctmgr): debug = 30 min, regular = 2 days.
# Requesting more than a QOS allows is rejected at submission (QOSMaxWallDurationPerJobLimit).
DEFAULT_WALLTIME = {"debug": "00:30:00", "regular": "02:00:00"}

# Workers per node for each stage, set by memory, not by core count.
# A Perlmutter CPU node has 512 GB. Measured peak memory for one dsp task
# on a 5000-sample file is ~10 GB (it holds the whole waveform table in
# memory), so 32 dsp workers use ~330 GB and leave headroom. Without a cap,
# Parsl starts one worker per 2 cores (128), which runs the node out of
# memory (WorkerLost / ManagerLost). raw streams the BIN file, so it is lighter.
DEFAULT_WORKERS_PER_NODE = {"raw": 64, "dsp": 32, "hit": 32}

DEFAULT_ACCOUNT = "m2676"
DEFAULT_CONDA_ENV = "legend-pipeline-parquet"


def make_local_config(max_threads=4):
    """Threads on this machine; no Slurm. For tests and small local runs."""
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor

    return Config(executors=[
        ThreadPoolExecutor(max_threads=max_threads, label="raw"),
        ThreadPoolExecutor(max_threads=max_threads, label="dsp"),
        ThreadPoolExecutor(max_threads=max_threads, label="hit"),
    ])


def make_config(nodes=1, account=DEFAULT_ACCOUNT, qos="regular", walltime=None,
                workers_per_node=None, conda_env=DEFAULT_CONDA_ENV, repo_dir=None):
    """Slurm (NERSC Perlmutter CPU) executors, one per stage.

    repo_dir defaults to this checkout, so each user's workers run their own
    copy of the code; conda_env is the environment the workers activate."""
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider

    if walltime is None:
        walltime = DEFAULT_WALLTIME.get(qos, "00:30:00")
    if workers_per_node is None:
        workers_per_node = DEFAULT_WORKERS_PER_NODE
    if repo_dir is None:
        repo_dir = REPO_ROOT

    # Compute-node workers don't inherit the login shell's directory or
    # PYTHONPATH; without this, `import pipeline` fails on the workers.
    worker_init = (
        f"cd {repo_dir} && "
        "module load conda && "
        f"conda activate {conda_env} && "
        f"export PYTHONPATH={repo_dir}:$PYTHONPATH"
    )

    def make_executor(label):
        return HighThroughputExecutor(
            label=label,
            cores_per_worker=2,
            max_workers_per_node=workers_per_node[label],
            provider=SlurmProvider(
                # keyword, not positional: SlurmProvider's first positional
                # argument is `partition`, so a positional qos is silently
                # ignored and jobs land in debug (30 min cap).
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

    return Config(executors=[make_executor("raw"), make_executor("dsp"), make_executor("hit")])
