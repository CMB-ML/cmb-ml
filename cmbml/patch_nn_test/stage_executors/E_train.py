import logging

from tqdm import tqdm
import csv

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from cmbml.core.asset_handlers.handler_npymap import NumpyMap
from cmbml.core.asset_handlers.appending_csv_handler import AppendingCsvHandler
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TrainCMBPrePatchDataset
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
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        # in_norm_handler: Config

        # self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.train.device)
        self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        self.batch_size = cfg.model.patch_nn.train.batch_size
        self.learning_rate = cfg.model.patch_nn.train.learning_rate
        self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]
        self.extra_check = cfg.model.patch_nn.train.extra_check
        self.checkpoint = cfg.model.patch_nn.train.checkpoint_every

        self.num_workers = cfg.model.patch_nn.train.num_loader_workers

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        loss_record_headers = ['Epoch', 'Training Loss', 'Validation Loss']
        self.out_loss_record.write(data=loss_record_headers)

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        model = model.to(self.device)

        # TODO: Potentially confusing... the order is determined 
        #       by the order in the pipeline yaml. In other contexts 
        #       this makes more sense.
        train_split = self.splits[0]
        valid_split = self.splits[1]

        # TODO: Dataset for validation (lower priority)
        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
            )

        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
            )

        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # TODO: Add mechanics for resuming training (code present in CMBNNCS)
        start_epoch = 0
        n_epoch_digits = len(str(self.n_epochs))

        all_train_loss = []
        all_valid_loss = []

        for epoch in range(start_epoch, self.n_epochs):
            # Training
            with tqdm(train_dataloader, desc=f"Ep {epoch + 1:<{n_epoch_digits}}", postfix={'Loss': 0}) as pbar:
                model.train()
                train_loss = 0
                for train_features, train_label in pbar:
                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    optimizer.zero_grad()
                    output = model(train_features)
                    batch_train_loss = loss_function(output, train_label)
                    batch_train_loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'Loss': batch_train_loss.item()})
                    train_loss += batch_train_loss.item()
                train_loss /= len(train_dataloader)
                all_train_loss.append(train_loss)
            logger.info(f"Epoch {epoch:<{n_epoch_digits}} Training loss: {train_loss:.02e}")

            # Validation
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for valid_features, valid_label in valid_dataloader:
                    valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                    valid_label = valid_label.to(device=self.device, dtype=self.dtype)
                    output = model(valid_features)
                    valid_loss += loss_function(output, valid_label).item()
                valid_loss /= len(valid_dataloader)
                all_valid_loss.append(valid_loss)
            logger.info(f"Epoch {epoch:<{n_epoch_digits}} Validation loss: {valid_loss:.02e}")

            self.out_loss_record.append([epoch + 1, train_loss, valid_loss])

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

    def set_up_dataset(self, split: Split) -> None:
        cmb_path_template = self.make_fn_template(split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(split, self.in_obs_assets)

        dataset = TrainCMBPrePatchDataset(
            n_sims = split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            label_handler=self.in_cmb_asset.handler,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
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
