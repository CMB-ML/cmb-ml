import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseDeepSphereModelExecutor
from cmbml.core import Split, Asset

from cmbml.core.asset_handlers import (
    PyTorchModel,
    HealpyMap,
    Config,
    AppendingCsvHandler
    )

from cmbml.deepsphere_unet.dataset import TrainCMBMapDataset


logger = logging.getLogger(__name__)


class DeterministicTrainingExecutor(BaseDeepSphereModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_best_epoch: Asset = self.assets_out["best_epoch"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        best_epoch_handler: Config
        loss_record_handler: AppendingCsvHandler

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_norm: Asset = self.assets_in["dataset_stats"]  # TODO: Does removing this line break anything?
        in_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        model_precision = 'float'
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.deepsphere.train.device)
        if self.device == "mps": # MPS is not supported for sparse models
            logger.info(f"MPS is not supported for sparse models. Using CPU.")
            self.choose_device("cpu")

        self.lr = cfg.model.deepsphere.train.learning_rate
        self.n_epochs = cfg.model.deepsphere.train.n_epochs
        self.batch_size = cfg.model.deepsphere.train.batch_size
        self.checkpoint = cfg.model.deepsphere.train.checkpoint_every
        self.extra_check = cfg.model.deepsphere.train.extra_check

        self.restart_epoch = cfg.model.deepsphere.train.restart_epoch


    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        dets_str = ', '.join([str(k) for k in self.instrument.dets.keys()])
        logger.info(f"Creating model using detectors: {dets_str}")

        logger.info(f"Using static learning rate.")
        logger.info(f"Learning rate is {self.lr}")
        logger.info(f"Number of epochs is {self.n_epochs}")
        logger.info(f"Batch size is {self.batch_size}")
        logger.info(f"Checkpoint every {self.checkpoint} iterations")
        logger.info(f"Extra check is set to {self.extra_check}")

        train_split = None
        valid_split = None
        for split in self.splits:
            if split.name == "Train":
                train_split = split
            elif split.name == "Valid":
                valid_split = split
        
        assert train_split is not None, "Train split not found, add train split in pipeline configuration file"
    
        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            )
        
        logger.info(f"Inspecting data for {train_split.name} split: ")
        self.inspect_data(train_dataloader)

        if valid_split is not None:
            valid_dataset = self.set_up_dataset(valid_split)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
            logger.info(f"Inspecting data for {valid_split.name} split: ")
            self.inspect_data(valid_dataloader)
            loss_record_headers = ["epoch", "train_loss", "valid_loss"]
        else:
            logger.info(f"No validation split found. Training without validation.")
            logger.info(f"This is not recommended. Consider adding a validation split in the pipeline configuration file.")
            valid_dataloader = None
            loss_record_headers = ["epoch", "train_loss"]

        model = self.make_model().to(self.device)
        
        logger.debug(f"Testing model output: ")
        self.try_model(model)

        
        self.out_loss_record.write(data=loss_record_headers)

        lr = self.lr
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(start_epoch, self.n_epochs):
            combined_loss = 0.0
            model.train()
            train_loss = self.train_loop(model, train_dataloader, optimizer, loss_function)
            if valid_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    valid_loss = self.validate(model, valid_dataloader, loss_function)
                self.out_loss_record.append(data=[epoch + 1, train_loss, valid_loss])
                combined_loss = 0.8 * valid_loss + 0.2 * train_loss
                logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss}, Valid Loss: {valid_loss}, Combined Loss: {combined_loss}")
            else:
                combined_loss = train_loss
                logger.info(f"Epoch {epoch + 1} Train Loss: {train_loss}, Combined Loss: {combined_loss}")
                self.out_loss_record.append(data=[epoch + 1, train_loss])
            
            if combined_loss < best_loss:
                best_loss = combined_loss
                best_epoch = epoch + 1
                with self.name_tracker.set_context("epoch", best_epoch):
                    res = {"best_epoch": best_epoch, "best_loss": best_loss}
                    self.out_best_epoch.write(data=res)
                    self.out_model.write(model=model, optimizer=optimizer, epoch=best_epoch, loss=best_loss)
                continue
            
            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         epoch=epoch + 1,
                                         loss=combined_loss)

    def train_loop(self, model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module) -> float:
        """Runs the training loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to train
            dataloader (DataLoader): Training Data
            optimizer (torch.optim.Optimizer): Optimizer
            loss_function (torch.nn.Module): Loss

        Returns:
            float: training loss for the epoch
        """

        epoch_loss = 0.0
        batch_n = 0
        batch_loss = 0
        with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
            for train_features, train_label in pbar:
                batch_n += 1

                train_features = train_features.to(device=self.device, dtype=self.dtype)
                train_label = train_label.to(device=self.device, dtype=self.dtype)

                optimizer.zero_grad()
                output = model(train_features)
                loss = loss_function(output, train_label)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

                pbar.set_postfix({f'Loss for {batch_n}/{len(dataloader)}': loss.item() / self.batch_size})

                epoch_loss += batch_loss
            epoch_loss /= len(dataloader.dataset)
        return epoch_loss
    
    def validate(self, model: torch.nn.Module, dataloader: DataLoader, loss_function: torch.nn.Module) -> float:
        """Runs the validation loop for a single epoch.

        Args:
            model (torch.nn.Module): Model to validate
            dataloader (DataLoader): Validation Data
            loss_function (torch.nn.Module): Loss

        Returns:
            float: validation loss for the epoch
        """
        epoch_loss = 0.0
        batch_n = 0
        batch_loss = 0
        with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
            for valid_features, valid_label in pbar:
                batch_n += 1

                valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                valid_label = valid_label.to(device=self.device, dtype=self.dtype)

                output = model(valid_features)
                loss = loss_function(output, valid_label)
                batch_loss += loss.item()

                pbar.set_postfix({f'Loss for {batch_n}/{len(dataloader)}': loss.item() / self.batch_size})

                epoch_loss += batch_loss
            epoch_loss /= len(dataloader.dataset)
        return epoch_loss

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            label_handler=HealpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=HealpyMap()
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}") # Should be (batch_size, npix, n_map_fields)
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
