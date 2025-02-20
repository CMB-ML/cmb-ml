import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import (
    PyTorchModel, 
    NumpyMap, 
    AppendingCsvHandler
    )
from cmbml.demo_patch_nn.dataset import TrainCMBPrePatchDataset
from cmbml.demo_patch_nn.dummy_model import SimpleUNetModel
from cmbml.demo_patch_nn.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class TrainingExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_model: Asset = self.assets_in["model"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_model_handler: PyTorchModel

        self.dtype         = self.dtype_mapping[cfg.model.cenn.dtype]  # TODO: Ensure this is used

        self.choose_device(cfg.model.cenn.train.device)  # See parent class
        self.batch_size    = cfg.model.cenn.train.batch_size
        self.num_workers   = cfg.model.cenn.train.num_loader_workers
        self.learning_rate = cfg.model.cenn.train.learning_rate
        self.n_epochs      = cfg.model.cenn.train.n_epochs
        self.restart_epoch = cfg.model.cenn.train.restart_epoch
        self.checkpoint    = cfg.model.cenn.train.checkpoint_every
        self.extra_check   = cfg.model.cenn.train.extra_check

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        loss_record_headers = ['Epoch', 'Training Loss', 'Validation Loss']
        self.out_loss_record.write(data=loss_record_headers)

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        model = model.to(self.device)

        # Potentially confusing... the order is determined by the order of
        #   splits for this stage in the pipeline yaml.
        train_split = self.splits[0]
        valid_split = self.splits[1]

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

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer, 
                                                #  scheduler=scheduler
                                                 )
            if start_epoch == "init":
                start_epoch = 0
        else:
            logger.info(f"Starting new model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
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
            logger.info(f"Epoch {epoch + 1:<{n_epoch_digits}} Training loss: {train_loss:.02e}")

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
            logger.info(f"Epoch {epoch + 1:<{n_epoch_digits}} Validation loss: {valid_loss:.02e}")

            self.out_loss_record.append([epoch + 1, train_loss, valid_loss])

            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                        #  scheduler=scheduler,
                                         epoch=epoch + 1)

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
            feature_handler=self.in_obs_assets.handler
            # Transform already happened in preprocessing
            )
        return dataset

    def get_patch_dict(self):
        patch_dict = self.in_all_p_ids_asset.read()
        patch_dict = patch_dict["patch_ids"]
        return patch_dict
