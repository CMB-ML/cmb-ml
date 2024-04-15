import numpy as np

from omegaconf import DictConfig, OmegaConf

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Split,
    Asset
)

class DirectoryExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = "create-directories"
        super().__init__(cfg, experiment)

    def execute(self) -> None:
        super().execute()
    
    def process_split(self, split: Split) -> None:
        for _, asset in self.assets_out.items():
            path = asset.path
            # print(path)
            asset.write(path)
    

            

