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


class ViewExecutor(BaseStageExecutor):
    def __init__(self,
                 cfg: DictConfig,
                 experiment: ExperimentParameters) -> None:
        # TODO: cfg_scope_str is not intuitive. Better to replace it.
        # The following string must match the pipeline yaml
        self.stage_str = "view"
        super().__init__(cfg, experiment)

        self.in_cmb_raw: Asset = self.assets_in["cmb_map"]
        self.in_obs_raw: Asset = self.assets_in["obs_maps"]
        self.in_cmb_pred: Asset = self.assets_in["cmb_map_post"]
        self.model_epochs = cfg.modelling.view.epoch
        self.map_field = 0
        self.current_epoch = None

        use_det_by_idx = 3

        channels: List[str] = ["cmb", *self.experiment.detector_freqs]
        self.use_obs_det = channels[use_det_by_idx]

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Executing ViewExecutor execute()")
        for epoch in self.model_epochs:
            self.current_epoch = epoch
            with self.name_tracker.set_context("epoch", epoch):
                for split in self.splits:
                    with self.name_tracker.set_context("split", split.name):
                        self.process_split(split)
                    break

    def process_split(self, 
                      split: Split) -> None:
        # logger.info(f"Executing ViewExecutor process_split() for split: {split.name} at {self.name_tracker.context["epoch"]}.")
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                logger.info(f"Executing ViewExecutor process_split() for: {split.name}:{sim} at {self.name_tracker.context['epoch']}.")
                self.process_sim()
            break

    def process_sim(self) -> None:
        raw_cmb = self.in_cmb_raw.read()[self.map_field]
        pred_cmb = self.in_cmb_pred.read()[self.map_field]
        all_raw_obs = self.in_obs_raw.read()
        raw_obs = all_raw_obs[self.use_obs_det][self.map_field]

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,5))

        plot_settings = dict(
            hold=True,
            min=raw_cmb.min(),
            max=raw_cmb.max(),
            xsize=800,
            reso=3
        )

        plt.axes(ax1)
        # hp.mollview(raw_cmb, **plot_settings)
        hp.gnomview(raw_obs, **plot_settings)
        plt.title(f"Observation at {self.use_obs_det} GHz")

        plt.axes(ax2)
        # hp.mollview(raw_cmb, **plot_settings)
        hp.gnomview(raw_cmb, **plot_settings)
        plt.title(f"Realization")

        plt.axes(ax3)
        # hp.mollview(pred_cmb, **plot_settings)
        hp.gnomview(pred_cmb, **plot_settings)
        plt.title(f"Prediction")

        epoch = self.name_tracker.context["epoch"]
        split = self.name_tracker.context["split"]
        sim_num = self.name_tracker.context["sim_num"]
        fig.suptitle(f"Comparison of Simulation {split}:{sim_num} for model at Epoch {epoch}")
        plt.show()
