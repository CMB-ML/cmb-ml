import logging

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.deepsphere_unet.dataset import TestCMBMapDataset
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel  # Import for typing hint
from .pytorch_model_base_executor import BaseDeepSphereModelExecutor
from cmbml.core.asset_handlers import HealpyMap            # Import for typing hint


logger = logging.getLogger(__name__)


class PredictionExecutor(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_model: Asset = self.assets_in["model"]
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        model_precision = 'float'
        self.dtype = self.dtype_mapping[model_precision]

        self.choose_device(cfg.model.deepsphere.predict.device)
        if self.device == "mps": # MPS is not supported for sparse models
            logger.info(f"MPS is not supported for sparse models. Using CPU.")
            self.choose_device("cpu")
        
        self.batch_size = cfg.model.deepsphere.predict.batch_size

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")

        for model_epoch in self.model_epochs:
            logger.debug(f"Making predictions based on epoch {model_epoch}")
            model = self.make_model()
            with self.name_tracker.set_context("epoch", model_epoch):
                self.in_model.read(model=model, epoch=model_epoch)
            model.eval().to(self.device)
            for split in self.splits:
                context = dict(split=split.name, epoch=model_epoch)
                with self.name_tracker.set_contexts(contexts_dict=context):
                        self.process_split(model, split)

    def process_split(self, model, split):
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
            )

        with torch.no_grad():
            for features, idcs in tqdm(dataloader):
                features_prepped = features.to(device=self.device, dtype=self.dtype)
                predictions = model(features_prepped)
                for pred, idx in zip(predictions, idcs):
                    with self.name_tracker.set_context("sim_num", idx.item()):
                        pred_npy = pred.detach().cpu().numpy()
                        self.out_cmb_asset.write(data=pred_npy)

    def set_up_dataset(self, template_split: Split) -> None:
        # We create a dataset for each split instead of a dataset that covers all
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TestCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=HealpyMap()
            )
        return dataset
