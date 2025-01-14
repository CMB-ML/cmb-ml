import logging

from tqdm import tqdm

# import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
from cmbml.patch_nn_test.dataset import TrainCMBMap2PatchDataset
from cmbml.patch_nn_test.dummy_model import SimpleUNetModel
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.patch_nn_test.utils.minmax_scale import MinMaxScaler


logger = logging.getLogger(__name__)


class TrainingNoPreprocessExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_full_map")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]
        self.in_model: Asset = self.assets_in["model"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config
        in_lut_handler: NumpyMap

        self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.train.device)
        self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        self.batch_size = cfg.model.patch_nn.train.batch_size
        self.learning_rate = 0.0002
        self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]
        self.extra_check = cfg.model.patch_nn.train.extra_check
        self.checkpoint = cfg.model.patch_nn.train.checkpoint_every

        self.scaling = cfg.model.patch_nn.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)
        self.extrema = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.extrema = self.get_extrema()

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        model = model.to(self.device)

        template_split = self.splits[0]

        # TODO: Dataset for validation (lower priority)
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
            )

        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # TODO: Add mechanics for resuming training (code present in CMBNNCS)
        start_epoch = 0

        for epoch in range(start_epoch, self.n_epochs):
            # batch_n = 0
            with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
                for train_features, train_label, sim_idx, p_idx in pbar:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()
                    output = model(train_features)
                    loss = loss_function(output, train_label)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'Loss': loss.item()})

                # TODO: Add validation loss (lower priority)
                # TODO: Add TensorBoard logging (lowest priority)

            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                        #  scheduler=scheduler,
                                         epoch=epoch + 1)
                                        #  loss=epoch_loss)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

    def get_extrema(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.extrema = self.in_norm_file.read()

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        with self.name_tracker.set_context("split", template_split.name):
            which_patch_dict = self.get_patch_dict()

        transform = None
        if self.scaling == "minmax":
            vmins = np.array([self.extrema[f]["I"]["vmin"].value for f in self.instrument.dets.keys()])
            vmaxs = np.array([self.extrema[f]["I"]["vmax"].value for f in self.instrument.dets.keys()])
            transform = MinMaxScaler(vmins=vmins, vmaxs=vmaxs)

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
            features_transform=transform
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
