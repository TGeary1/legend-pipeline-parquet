
#Local thread pools for testing without Slurm
def make_local_config(max_threads=4):
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor

    return Config(executors=[
        ThreadPoolExecutor(max_threads=max_threads, label="raw"),
        ThreadPoolExecutor(max_threads=max_threads, label="dsp"),
        ThreadPoolExecutor(max_threads=max_threads, label="hit"),
    ])



#Executors on Perlmutter nodes
def make_config(max_workers_per_node=16, nodes=1, account="m2676", qos="debug"):
    from parsl.config import Config
    from parsl.providers import SlurmProvider
    from parsl.launchers import SrunLauncher
    from parsl.executors import HighThroughputExecutor

    venv_activate = "source /global/path/to/legend-pipeline-parquet/venv/bin/activate"

    def make_executor(label):
        return HighThroughputExecutor(
            label=label,
            cores_per_worker=2,
            provider=SlurmProvider(
                qos,
                account=account,
                nodes_per_block=nodes,
                scheduler_options="#SBATCH -C cpu",
                worker_init='module load conda && conda activate legend-pipeline-parquet',
                launcher=SrunLauncher(overrides="-c 128"),
                walltime="00:30:00",
                cmd_timeout=120,
            ),
        )

    return Config(executors=[
        make_executor("raw"),
        make_executor("dsp"),
        make_executor("hit"),
    ])
