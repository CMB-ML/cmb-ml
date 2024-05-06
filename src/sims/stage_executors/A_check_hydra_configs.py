import logging
from omegaconf import DictConfig

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigCheckerExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = 'check-configs'
        super().__init__(cfg, experiment)

    def execute(self) -> None:
        assert self.cfg.simulation.nside_sky > self.experiment.nside, "nside of sky should be greater than nside of target output by at least a factor of 2"
        logger.debug("No conflict in Hydra Configs found.")
