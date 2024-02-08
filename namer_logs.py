import logging
from pathlib import Path

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


logger = logging.getLogger(__name__)


class LogsNamer:
    def __init__(self, 
                 conf: DictConfig,
                 hydra_config: HydraConfig,
                 dataset_path: Path) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.hydra_run_root = Path(hydra_config.runtime.cwd)
        self.hydra_run_dir = hydra_config.run.dir
        self.scripts_subdir = conf.file_system.subdir_for_log_scripts
        self.dataset_root = dataset_path
        self.dataset_logs_dir = conf.file_system.subdir_for_logs

    @property
    def hydra_path(self) -> Path:
        return self.hydra_run_root / self.hydra_run_dir

    @property
    def hydra_scripts_path(self) -> Path:
        return self.hydra_path / self.scripts_subdir

    @property
    def dataset_logs_path(self) -> Path:
        # Preserve source directory structure; 
        #    do NOT overwrite logs if some dataset has already been created
        logs_dataset_path = self.dataset_root / self.dataset_logs_dir
        logs_dataset_path = logs_dataset_path / self.hydra_run_dir
        return logs_dataset_path
