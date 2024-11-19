import logging
from omegaconf import DictConfig

import numpy as np

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigSimsCheckerExecutor(BaseStageExecutor):
    """
    A stage executor class that checks Hydra configs for any issues.

    Attributes:
        cfg (DictConfig): The Hydra config to check.

    Methods:
        execute(): Check for any conflicts in the Hydra configs.
        check_units(): Check if the units are supported.
        check_noise_yaml(): Check if a detector is in the noise yaml.
        check_simulation_yaml(): Check if the simulation yaml has any conflicts.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='check_hydra_configs')
        # TODO: Use logging import configs logic to check for duplicate pipeline stage names (unless hydra crashes on this? Investigate.)
        # TODO: Remove duplication of effort in src.core check_hydra_configs
        self.issues = []

    def execute(self) -> None:
        """
        Execute the HydraConfigSimsChecker stage to check for
        any issues in the Hydra configs.

        Raises:
            ValueError: If any conflicts are found in the Hydra configs.
        """
        self.check_noise_yaml()
        self.check_simulation_yaml()
        for issue in self.issues:
            logger.warning(issue)
        if len(self.issues) > 0:
            raise ValueError("Conflicts found in hydra configs.")
        logger.debug("No conflict in Hydra Configs found.")

    def check_noise_yaml(self) -> None:
        """
        Ensure all detectors' noise files are in the simulation.noise yaml.
        """
        if self.cfg.model.sim.noise.do_cache is True:
            for freq in self.cfg.scenario.detector_freqs:
                if freq not in self.cfg.model.sim.noise.src_files:
                    self.issues.append(f"Detector {freq} not in simulation.noise yaml.")

    def check_simulation_yaml(self) -> None:
        """
        Check if the simulation yaml for some known potential conflicts.
        """
        nside_out = self.cfg.scenario.nside
        nside_sky_set = self.cfg.model.sim.get("nside_sky", None)
        nside_sky_factor = self.cfg.model.sim.get("nside_sky_factor", None)
        if nside_sky_set is None and nside_sky_factor is None:
            self.issues.append("Either nside_sky or nside_sky_factor must be set in simulation yaml.")
        elif nside_sky_set is not None and nside_sky_factor is not None:
            self.issues.append("Either nside_sky or nside_sky_factor must be set in simulation yaml.")
        else:
            nside_sky = nside_sky_set if nside_sky_set else nside_out * nside_sky_factor
            if nside_sky > 8192:
                self.issues.append("PySM3 does not support resolutions above 8192.")
            elif nside_sky == 8192:
                # Largest supported. If this is used, the nside_out can be 8192. Not for the faint of heart!
                pass
            elif nside_sky < 2 * nside_out:
                self.issues.append("nside of sky should be greater than nside of target output by at least a factor of 2 in simulation yaml")
            if not self._is_power_of_two(nside_sky):
                self.issues.append("nside_sky must be a power of 2")
            if not self._is_power_of_two(nside_out):
                self.issues.append("nside_out must be a power of 2")

    @staticmethod
    def _is_power_of_two(n):
        if n <= 0:
            return False
        return (n & (n - 1)) == 0
