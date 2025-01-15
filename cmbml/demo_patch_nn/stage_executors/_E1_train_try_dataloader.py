import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
from cmbml.demo_patch_nn.dataset import TrainCMBMap2PatchDataset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.demo_patch_nn.utils.display_help import show_patch
from cmbml.demo_patch_nn.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.demo_patch_nn.utils.minmax_scale import MinMaxScaler


logger = logging.getLogger(__name__)


class TrainingTryDataloaderExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_no_preprocess")

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]

        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config
        in_lut_handler: NumpyMap
        in_all_p_ids_handler: Config

        self.nside_patch = cfg.model.patches.nside_patch

        self.batch_size = 4  # Batch size can be very large; for demo purposes, we use 4
        # self.batch_size = cfg.model.patch_nn.train.batch_size

        self.scaling = cfg.model.patch_nn.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)
        self.dataset_stats = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.load_dataset_stats()

        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # TODO: Change to True (when using the full dataset)
            )

        batch_n = 0
        for train_features, train_label, sim_idx, p_idx in dataloader:
            logger.debug(f"train label shape: {train_label.shape}", flush=True)
            logger.debug(f"train features shape: {train_features.shape}", flush=True)
            for i in range(self.batch_size):
                show_patch(train_label[i, 0, :], train_features[i, :], 
                            f"Batch {batch_n}, Sample {i}, Train{sim_idx[i]:04d}, Patch {p_idx[i]}")
            if sim_idx[-1] >= 10 - self.batch_size:
                # I have 10 sims, so this will show the last *full* batch
                break
            batch_n += 1

    def load_dataset_stats(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        with self.name_tracker.set_context("split", template_split.name):
            which_patch_dict = self.get_patch_dict()

        features_transform = None
        if self.scaling == "minmax":
            vmins = np.array([self.dataset_stats[f]["I"]["vmin"].value for f in self.instrument.dets.keys()])
            vmaxs = np.array([self.dataset_stats[f]["I"]["vmax"].value for f in self.instrument.dets.keys()])
            features_transform = MinMaxScaler(vmins=vmins, vmaxs=vmaxs)

            vmins = self.dataset_stats["cmb"]["I"]["vmin"].value
            vmaxs = self.dataset_stats["cmb"]["I"]["vmax"].value
            labels_transform = MinMaxScaler(vmins=vmins, vmaxs=vmaxs)

        dataset = TrainCMBMap2PatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            label_handler=self.in_cmb_asset.handler,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            which_patch_dict=which_patch_dict,
            lut=self.in_lut_asset.read(),
            features_transform=features_transform,
            labels_transform=labels_transform
            )
        return dataset

    def get_patch_dict(self):
        patch_dict = self.in_all_p_ids_asset.read()
        patch_dict = patch_dict["patch_ids"]
        return patch_dict

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1] * train_features.size()[-2]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
