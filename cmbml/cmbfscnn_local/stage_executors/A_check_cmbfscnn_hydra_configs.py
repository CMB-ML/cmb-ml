import logging
from omegaconf import DictConfig

import numpy as np

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigCMBFSCNNCheckerExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="check_hydra_configs")
        # TODO: Use logging import configs logic to check for duplicate pipeline stage names
        self.issues = []
    
    def execute(self) -> None:
        self.check_scenario_yaml()
        for issue in self.issues:
            logger.warning(issue)
        if len(self.issues) > 0:
            raise ValueError("Conflicts found in Hydra configs.")
        logger.debug("No conflict in Hydra configs found.")

    def check_scenario_yaml(self) -> None:
        if self.cfg.scenario.map_fields != "I":
            self.issues.append("Currently, the only mode supported for training CMBFSCNN is Temperature. In the scenario yaml, change map_fields.")