import logging

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from omegaconf import DictConfig

from core import (
    # BaseStageExecutor, 
    Split,
    ExperimentParameters,
    Asset,
    HealpyMap, ManyHealpyMaps
    )

from ..handler_npymap import ManyNumpyMaps
from ..dataset import TestCMBMapDataset
from .pytorch_model_base_executor import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class PredictionExecutor(BasePyTorchModelExecutor):
    def __init__(self,
                 cfg: DictConfig,
                 experiment: ExperimentParameters) -> None:
        self.stage_str = "predict"
        super().__init__(cfg, experiment)

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]

        self.in_model: Asset = self.assets_in["model"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]

        self.choose_device(cfg.training.predict.device)

        self.model_epochs = cfg.training.predict.epoch
        self.batch_size = cfg.training.predict.batch_size

    def execute(self) -> None:
        logger.debug(f"Executing Prediction Executor execute()")

        for model_epoch in self.model_epochs:
            logger.debug(f"Making predictions based on epoch {model_epoch}")
            model = self.make_model()
            self.in_model.read(model, model_epoch)
            model.eval()
            model.to(self.device)

            with self.name_tracker.set_context("epoch", model_epoch):

                for split in tqdm(self.splits):
                    with self.name_tracker.set_context("split", split):
                        self.process_split(model, split)

    def set_up_dataset(self, template_split: Split) -> None:
        # For now, we consider only the most likely case that there's a single split
        #    being used for the creation of any PyTorch Dataset.
        # We cannot use the assets as I've created them, which use another object to
        #    track filenames.
        # The only thing that changes within a CMBMapDataset's filepaths is the sim #
        # So I create a single template string for each of the maps.
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        # TODO: Use parameter for ManyNumpyMaps
        dataset = TestCMBMapDataset(
            n_sims = template_split.n_sims,
            detectors = self.experiment.detector_freqs,
            feature_path_template=obs_path_template,
            feature_handler=ManyNumpyMaps(self.experiment)
            )
        return dataset

    def process_split(self, model, split):
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
            )

        with torch.no_grad():
            for features, idcs in dataloader:
                features_prepped = self.prep_data(features)
                predictions = model(features_prepped)
                for pred, idx in zip(predictions, idcs):
                    with self.name_tracker.set_context("sim_num", idx.item()):
                        # TODO: FIX
                        # logger.warning("Is this a hack?")
                        pred_npy = pred.detach().cpu().numpy()
                        # logger.debug(f"Writing {self.out_cmb_asset.path}")
                        self.out_cmb_asset.write(pred_npy)
