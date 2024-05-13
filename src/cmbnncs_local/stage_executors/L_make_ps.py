from typing import List, Dict
import logging

import healpy as hp
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from core import (
    BaseStageExecutor, 
    Split,
    ExperimentParameters,
    Asset
    )


logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)


class MakePSExecutor(BaseStageExecutor):
    def __init__(self,
                 cfg: DictConfig,
                 experiment: ExperimentParameters) -> None:
        logger.debug("Initializing MakePSExecutor")
        # TODO: cfg_scope_str is not intuitive. Better to replace it.
        # The following string must match the pipeline yaml
        self.stage_str = "make_ps"
        super().__init__(cfg, experiment)

        self.in_cmb_map: Asset = self.assets_in["cmb_map_post"]
        self.out_cmb_ps: Asset = self.assets_out["cmb_ps_post"]
        self.model_epochs = cfg.modelling.view.epoch

        self.map_field = 0

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Executing MakePSExecutor execute()")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)
            break

    def process_split(self, 
                      split: Split) -> None:
        # logger.info(f"Executing MakePSExecutor process_split() for split: {split.name} at {self.name_tracker.context["epoch"]}.")
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                logger.info(f"Executing MakePSExecutor process_split() for: {split.name}:{sim} at {self.name_tracker.context['epoch']}.")
                self.process_sim()
            break

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(epoch)

    def process_epoch(self) -> None:
        cmb_map = self.in_cmb_map.read()[self.map_field]
        cmb_ps = hp.anafast(cmb_map)