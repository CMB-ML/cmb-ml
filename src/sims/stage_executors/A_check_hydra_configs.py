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
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, experiment, stage_str='check_hydra_configs')

    def execute(self) -> None:
        assert self.cfg.simulation.nside_sky > self.experiment.nside, "nside of sky should be greater than nside of target output by at least a factor of 2"
        
        self.check_pipeline_yaml()
        logger.debug("No conflict in Hydra Configs found.")

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
            if 'assets_in' in stage_data:
                for asset_name, asset_data in stage_data['assets_in'].items():
                    # Check if asset_in has the required 'stage' key
                    if 'stage' not in asset_data:
                        raise ValueError(f"Asset '{asset_name}' in '{stage_name}' does not specify a 'stage'.")
                    # Check if the input asset was created in the specified stage's outputs
                    required_stage = asset_data['stage']
                    if required_stage not in outputs or asset_name not in outputs[required_stage]:
                        raise ValueError(f"Asset '{asset_name}' in '{stage_name}' cannot be found in outputs of stage '{required_stage}'.")
