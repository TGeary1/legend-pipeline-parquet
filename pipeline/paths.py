"""Repository locations, so config files are found no matter which directory
a script is run from (relative paths like "config/..." only work from the
repo root)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

DEFAULT_WINDOW_CONFIG = CONFIG_DIR / "dsp_window_configs.yaml"
DEFAULT_CALIBRATION = CONFIG_DIR / "spe_calibration.yaml"
DEFAULT_RUN_PHASES = CONFIG_DIR / "run_phases.yaml"
