import logging
from omegaconf import DictConfig

import numpy as np

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigCMBNNCSCheckerExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='check_hydra_configs')
        # TODO: Use logging import configs logic to check for duplicate pipeline stage names
        self.issues = []

    def execute(self) -> None:
        self.check_scenario_yaml()
        for issue in self.issues:
            logger.warning(issue)
        if len(self.issues) > 0:
            raise ValueError("Conflicts found in hydra configs.")
        logger.debug("No conflict in Hydra Configs found.")

    def check_scenario_yaml(self) -> None:
        if self.cfg.scenario.map_fields != "I":
            self.issues.append("Currently, the only mode supported for training CMBNNCS is Temperature. In the scenario yaml, change map_fields.")
        if 857 in self.cfg.scenario.detector_freqs:
            self.issues.append("Currently, CMBNNCS is more likely to fail when the 857 GHz Detector is included.")
        if 545 in self.cfg.scenario.detector_freqs:
            self.issues.append("Currently, CMBNNCS is more likely to fail when the 545 GHz Detector is included.")
