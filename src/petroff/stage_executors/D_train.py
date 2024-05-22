from typing import List, Callable

import logging

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import PetroffModelExecutor
from core import Split, Asset
from core.asset_handlers.asset_handlers_base import Config
from core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap
from core.pytorch_dataset import TrainCMBMapDataset
from core.pytorch_transform import TrainToTensor, train_remove_map_fields
from petroff.preprocessing.scale_methods_factory import get_scale_class
from petroff.preprocessing.pytorch_transform_pixel_reorder import ReorderTransform


logger = logging.getLogger(__name__)


class TrainingExecutor(PetroffModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_norm: Asset = self.assets_in["norm_file"]
        out_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config

        self.norm_data = None

        model_precision = cfg.model.petroff.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.petroff.train.device)

        self.lr = cfg.model.petroff.train.learning_rate
        self.n_epochs = cfg.model.petroff.train.n_epochs
        self.batch_size = cfg.model.petroff.train.batch_size
        self.checkpoint = cfg.model.petroff.train.checkpoint_every
        self.extra_check = cfg.model.petroff.train.extra_check
        self.scale_class = None
        self.set_scale_class(cfg)
        # self.num_workers = cfg.model.petroff.train.get("num_workers", 0)

        self.restart_epoch = cfg.model.petroff.train.restart_epoch

    def set_scale_class(self, cfg):
        scale_method = cfg.model.petroff.preprocess.scaling
        self.scale_class = get_scale_class(method=scale_method, 
                                           dataset="train", 
                                           scale="scale")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        # We use the first split as a template for setting up the dataset.
        #    This makes more sense in the context of inference over multiple
        #    different splits.
        dets_str = ', '.join([str(k) for k in self.instrument.dets.keys()])
        logger.info(f"Creating model using detectors: {dets_str}")

        logger.info(f"Using fixed learning rate.")
        logger.info(f"Learning rate is {self.lr}")
        logger.info(f"number of epochs is {self.n_epochs}")
        logger.info(f"batch size is {self.batch_size}")
        logger.info(f"checkpoint every {self.checkpoint} iterations")
        logger.info(f"extra check is set to {self.extra_check}")

        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )
        self.inspect_data(dataloader)

        model = self.make_model()
        with torch.no_grad():
            self.try_model(model)

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer)
            if start_epoch == "init":
                start_epoch = 0
        else:
            logger.info(f"Starting new model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
            start_epoch = 0

        for epoch in range(start_epoch, self.n_epochs):
            epoch_loss = 0.0
            batch_n = 0

            with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
                for train_features, train_label in pbar:
                    batch_n += 1

                    optimizer.zero_grad()
                    output = model(train_features)
                    loss = loss_function(output, train_label)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        epoch_loss += loss.item()
                        pbar.set_postfix({'Loss': loss.item() / self.batch_size})
                        print(loss.item() / self.batch_size)

            with torch.no_grad():
                epoch_loss /= len(dataloader.dataset)
                logger.info(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}')

                # Checkpoint every so many epochs
                if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                    with self.name_tracker.set_context("epoch", epoch + 1):
                        self.out_model.write(model=model,
                                            optimizer=optimizer,
                                            epoch=epoch + 1,
                                            loss=epoch_loss)

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        scale_factors = self.in_norm.read()
        dtype_transform = TrainToTensor(self.dtype, device="cpu")
        scale_map_transform = self.scale_class(all_map_fields=self.map_fields,
                                               scale_factors=scale_factors,
                                               device="cpu",
                                               dtype=self.dtype)
        device_transform = TrainToTensor(self.dtype, device=self.device)
        remove_map_field_transform = train_remove_map_fields
        pt_transforms = [
            dtype_transform,
            scale_map_transform,
            remove_map_field_transform,
            device_transform
        ]

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            read_to_nest=True,          # Because Petroff uses hierarchical format
            pt_xforms=pt_transforms,
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"TrainingExecutor.preview_data() Feature batch shape: {train_features.size()}")
        logger.info(f"TrainingExecutor.preview_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Data map resolution does not match configuration."
