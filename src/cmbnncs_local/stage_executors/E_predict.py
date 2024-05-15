import logging

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from omegaconf import DictConfig

from core import Split, Asset
from core.dataset import TestCMBMapDataset
from core.asset_handlers.handler_model_pytorch import PyTorchModel  # Import for typing hint
from .pytorch_model_base_executor import BaseCMBNNCSModelExecutor
from ..handler_npymap import NumpyMap             # Import for typing hint


logger = logging.getLogger(__name__)


class PredictionExecutor(BaseCMBNNCSModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: NumpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_model: Asset = self.assets_in["model"]
        in_obs_map_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.choose_device(cfg.model.cmbnncs.predict.device)
        self.batch_size = cfg.model.cmbnncs.predict.batch_size

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")

        for model_epoch in self.model_epochs:
            logger.debug(f"Making predictions based on epoch {model_epoch}")
            model = self.make_model()
            with self.name_tracker.set_context("epoch", model_epoch):
                self.in_model.read(model=model)
            model.eval().to(self.device)
            for split in self.splits:
                context = dict(split=split.name, epoch=model_epoch)
                with self.name_tracker.set_contexts(contexts_dict=context):
                        self.process_split(model, split)

    def set_up_dataset(self, template_split: Split) -> None:
        # We create a dataset for each split instead of a dataset that covers all
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TestCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            feature_path_template=obs_path_template,
            file_handler=NumpyMap()
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
            for features, idcs in tqdm(dataloader):
                features_prepped = self.prep_data(features)
                predictions = model(features_prepped)
                for pred, idx in zip(predictions, idcs):
                    with self.name_tracker.set_context("sim_num", idx.item()):
                        pred_npy = pred.detach().cpu().numpy()
                        self.out_cmb_asset.write(data=pred_npy)
