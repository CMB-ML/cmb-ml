# from typing import List, Dict
# import logging

# import numpy as np

# from omegaconf import DictConfig

# from ...core import (
#     BaseStageExecutor, 
#     Split,
#     ExperimentParameters,
#     Asset
#     )


# logger = logging.getLogger(__name__)


# class SerialPostprocessExecutor(BaseStageExecutor):
#     def __init__(self,
#                  cfg: DictConfig,
#                  experiment: ExperimentParameters) -> None:
#         logger.debug("Initializing Petroff PostprocessExecutor")
#         # TODO: cfg_scope_str is not intuitive. Better to replace it.
#         # The following string must match the pipeline yaml
#         self.stage_str = "postprocess"
#         super().__init__(cfg, experiment)

#         self.out_cmb_asset: Asset = self.assets_out["cmb_map_post"]

#         self.in_norm_file: Asset = self.assets_in["norm_file"]
#         self.in_cmb_asset: Asset = self.assets_in["cmb_map"]

#         self.model_epochs = cfg.modelling.postprocess.epoch

#     def execute(self) -> None:
#         # Remove this function
#         logger.debug(f"Running {self.__class__.__name__} execute() method.")
#         for epoch in self.model_epochs:
#             logger.debug(f"Executing PostprocessExecutor execute() for epoch: {epoch}")
#             with self.name_tracker.set_context("epoch", epoch):
#                 super().execute()

#     def process_split(self, 
#                       split: Split) -> None:
#         logger.info(f"Executing PostprocessExecutor process_split() for split: {split.name}.")
        
#         logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
#         extrema = self.in_norm_file.read()
#         for sim in split.iter_sims():
#             with self.name_tracker.set_context("sim_num", sim):
#                 self.process_sim(extrema)

#     def process_sim(self, 
#                     all_extrema) -> None:
#         in_cmb_map = self.in_cmb_asset.read()
#         normed_map = self.unnormalize_map_file(in_cmb_map, channel_extrema=all_extrema['cmb'])
#         self.out_cmb_asset.write(normed_map)

#     def unnormalize_map_file(self, 
#                            map_data: np.ndarray, 
#                            channel_extrema: Dict[str, Dict[str, float]]) -> List[np.ndarray]:
#         normed_map = []
#         for field_n in range(map_data.shape[0]):
#             field_char = self.experiment.map_fields[field_n]
#             field_data = map_data[field_n]
#             field_extrema = channel_extrema[field_char]
#             normed_map.append(self.normalize(field_data, field_extrema))

#         return normed_map

#     def normalize(self, in_map, extrema):
#         norm_min = extrema['min_val']
#         norm_max = extrema['max_val']

#         out_map = in_map * (norm_max - norm_min) + norm_min

#         return out_map
