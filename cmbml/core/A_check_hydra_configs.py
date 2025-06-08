import logging
from omegaconf import DictConfig

import numpy as np

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset
)

logger = logging.getLogger(__name__)


class HydraConfigCheckerExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='check_hydra_configs')
        # TODO: Use logging import configs logic to check for duplicate pipeline stage names
        self.issues = []

    def execute(self) -> None:
        # self.check_scenario_yaml()
        self.check_pipeline_yaml()
        for issue in self.issues:
            logger.warning(issue)
        if len(self.issues) > 0:
            issues = '\n'.join(self.issues)
            raise ValueError(f"Conflicts found in hydra configs. {issues}")
        logger.debug("No conflict in Hydra Configs found.")

    # def check_scenario_yaml(self) -> None:
    #     for freq in self.cfg.scenario.detector_freqs:
    #         if freq not in self.cfg.scenario.full_instrument:
    #             self.issues.append(f"Detector {freq} not in instrument list in scenario yaml.")

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
                orig_name = asset_data.get('orig_name', asset_name)
                if required_stage not in outputs:
                    self.issues.append(f"Asset '{asset_name}' in '{stage_name}' cites '{required_stage}', which does not exist in pipeline yaml.")
                elif orig_name not in outputs[required_stage]:
                    self.issues.append(f"Asset '{asset_name}' in '{stage_name}' cannot be found in outputs of stage '{required_stage}' in pipeline yaml.")

        for stage_name, stage_data in pipeline.items():
            if stage_data is None:
                continue
            if stage_data.get('make_stage_log', False) and 'dir_name' not in stage_data:
                self.issues.append(f"Stage {stage_name} has make_stage_log=True but no directory name in pipeline yaml.")
