import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import (
    Config,
    PyTorchModel, 
    HealpyMap,
    NumpyMap
    )
from cmbml.demo_patch_nn.dataset import TrainCMBMap2PatchDataset
from cmbml.demo_patch_nn.dummy_model import SimpleUNetModel
from cmbml.demo_patch_nn.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.demo_patch_nn.utils.minmax_scale import MinMaxScaler


logger = logging.getLogger(__name__)


class TrainingNoPreprocessExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_no_preprocess")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]
        self.in_model: Asset = self.assets_in["model"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config
        in_lut_handler: NumpyMap
        in_all_p_ids_handler: Config
        in_model_handler: PyTorchModel

        self.nside_patch = cfg.model.patches.nside_patch

        # self.choose_device(cfg.model.patch_nn.train.device)
        # self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        # self.batch_size = cfg.model.patch_nn.train.batch_size
        # self.learning_rate = 0.0002
        # self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]
        # self.extra_check = cfg.model.patch_nn.train.extra_check
        # self.checkpoint = cfg.model.patch_nn.train.checkpoint_every

        self.dtype         = self.dtype_mapping[cfg.model.patch_nn.dtype]  # TODO: Ensure this is used

        self.choose_device(cfg.model.patch_nn.train.device)  # See parent class
        self.batch_size    = cfg.model.patch_nn.train.batch_size
        self.num_workers   = cfg.model.patch_nn.train.num_loader_workers
        self.learning_rate = cfg.model.patch_nn.train.learning_rate
        self.n_epochs      = cfg.model.patch_nn.train.n_epochs
        self.restart_epoch = cfg.model.patch_nn.train.restart_epoch
        self.checkpoint    = cfg.model.patch_nn.train.checkpoint_every
        self.extra_check   = cfg.model.patch_nn.train.extra_check

        self.scaling = cfg.model.patch_nn.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)
        self.dataset_stats = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.load_dataset_stats()

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        model = model.to(self.device)

        template_split = self.splits[0]

        # TODO: Dataset for validation (low priority; see E_train.py for an example of how to implement)
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
            )

        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer, 
                                                 )
            if start_epoch == "init":
                start_epoch = 0
        else:
            logger.info(f"Starting new model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
            start_epoch = 0

        n_epoch_digits = len(str(self.n_epochs))

        for epoch in range(start_epoch, self.n_epochs):
            # batch_n = 0
            with tqdm(dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
                for train_features, train_label, sim_idx, p_idx in pbar:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()
                    output = model(train_features)
                    loss = loss_function(output, train_label)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'Loss': loss.item()})

            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         epoch=epoch + 1)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def load_dataset_stats(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()
            return self.dataset_stats

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        with self.name_tracker.set_context("split", template_split.name):
            which_patch_dict = self.get_patch_dict()

        features_transform = None
        labels_transform = None
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
