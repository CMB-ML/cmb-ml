import logging

from tqdm import tqdm

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.handler_npymap import NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TrainCMBMap2PatchDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.patch_nn_test.utils.display_help import show_patch
from cmbml.patch_nn_test.dummy_model import SimpleUNetModel
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class TrainingExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_full_map")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_lut_handler: NumpyMap
        # in_norm_handler: Config

        self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.train.device)
        self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        self.batch_size = cfg.model.patch_nn.train.batch_size
        self.learning_rate = 0.0002
        self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]
        self.extra_check = cfg.model.patch_nn.train.extra_check
        self.checkpoint = cfg.model.patch_nn.train.checkpoint_every

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        model = model.to(self.device)

        template_split = self.splits[0]

        # TODO: Include normalization (high priority, requires an executor to scan files. See Petroff method)
        # TODO: Preprocess dataset (cut into patches, normalize, save, etc.) (??? priority - this seems slow currently)
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

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        with self.name_tracker.set_context("split", template_split.name):
            which_patch_dict = self.get_patch_dict()

        dataset = TrainCMBMap2PatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            label_handler=self.in_cmb_asset.handler,
            # label_handler=HealpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            which_patch_dict=which_patch_dict,
            nside_obs=self.nside,
            nside_patches=self.nside_patch,
            lut=self.in_lut_asset.read(),
            # feature_handler=HealpyMap()
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