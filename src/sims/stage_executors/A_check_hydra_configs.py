import logging
from omegaconf import DictConfig

from ...core import (
    BaseStageExecutor,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigCheckerExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='check_hydra_configs')
        self.issues = []

    def execute(self) -> None:
        self.check_scenario_yaml()
        self.check_noise_yaml()
        self.check_simulation_yaml
        self.check_pipeline_yaml()
        for issue in self.issues:
            logger.warning(issue)
        if len(self.issues) > 0:
            raise ValueError("Conflicts found in hydra configs.")
        logger.debug("No conflict in Hydra Configs found.")

    def check_units(self) -> None:
        if self.cfg.scenario.units != "K_CMB":
            self.issues.append("Currently, the only supported units are K_CMB. Hardcoding for this exists throughout, but will be removed in a future version.")

    def check_scenario_yaml(self) -> None:
        for freq in self.cfg.scenario.detector_freqs:
            if freq not in self.cfg.scenario.full_instrument:
                self.issues.append(f"Detector {freq} not in instrument list in scenario yaml.")

    def check_noise_yaml(self) -> None:
        for freq in self.cfg.scenario.detector_freqs:
            if freq not in self.cfg.simulation.noise.src_files:
                self.issues.append(f"Detector {freq} not in simulation.noise yaml.")

    def check_simulation_yaml(self) -> None:
        nside_out = self.cfg.scenario.nside
        nside_sky_set = self.cfg.simulation.get("nside_sky", None)
        nside_sky_factor = self.cfg.simulation.get("nside_sky_factor", None)
        if nside_sky_set is None and nside_sky_factor is None:
            self.issues.append("Either nside_sky or nside_sky_factor must be set in simulation yaml.")
        elif nside_sky_set is not None and nside_sky_factor is not None:
            self.issues.append("Either nside_sky or nside_sky_factor must be set in simulation yaml.")
        else:
            nside_sky = nside_sky_set if nside_sky_set else nside_out * nside_sky_factor
            if nside_sky > 8192:
                self.issues.append("PySM3 does not support these resolutions")
            elif nside_sky == 8192:
                # Largest supported, but cannot use higher scale map. Not for the faint of heart!
                pass
            elif nside_sky < 2 * nside_out:
                self.issues.append("nside of sky should be greater than nside of target output by at least a factor of 2 in simulation yaml")

    def check_pipeline_yaml(self) -> None:
        pipeline = self.cfg.pipeline
        outputs = {}
        # Build list of assets_out for all stages
        for stage_name, stage_data in pipeline.items():
            if stage_data is None:
                continue
            outputs[stage_name] = set()
            if 'assets_out' in stage_data:
                outputs[stage_name].update(stage_data['assets_out'].keys())

        # Check each input in all stages
        for stage_name, stage_data in pipeline.items():
            if stage_data is None:
                continue
            if 'assets_in' not in stage_data:
                continue
            for asset_name, asset_data in stage_data['assets_in'].items():
                # Check if asset_in has the required 'stage' key
                if 'stage' not in asset_data:
                    self.issues.append(f"Asset '{asset_name}' in '{stage_name}' does not specify a 'stage' in pipeline yaml.")
                # Check if the input asset was created in the specified stage's outputs
                required_stage = asset_data['stage']
                if required_stage not in outputs or asset_name not in outputs[required_stage]:
                    self.issues.append(f"Asset '{asset_name}' in '{stage_name}' cannot be found in outputs of stage '{required_stage}' in pipeline yaml.")
