import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from core import (
    Split,
    Asset
    )


from .pytorch_model_base_executor import BasePyTorchModelExecutor
from ..dataset import LabelledCMBMapDataset
from ..handler_model_pytorch import PyTorchModel  # Import for typing hint
from ..handler_npymap import NumpyMap             # Import for typing hint

logger = logging.getLogger(__name__)


class TrainingExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap

        self.choose_device(cfg.model.cmbnncs.train.device)

        self.n_epochs = cfg.model.cmbnncs.train.n_epochs
        self.batch_size = cfg.model.cmbnncs.train.batch_size
        self.checkpoint = cfg.model.cmbnncs.train.checkpoint_every
        self.output_every = cfg.model.cmbnncs.train.output_every
        self.extra_check = cfg.model.cmbnncs.train.extra_check
        self.lr_init = cfg.model.cmbnncs.train.learning_rate
        self.lr_final = cfg.model.cmbnncs.train.learning_rate_min
        self.repeat_n = cfg.model.cmbnncs.train.repeat_n

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        # Code smell? Consider, instead of a template_split, 
        #   a dataset object that is common to all splits, 
        #   containing information needed for set_up_dataset()
        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )

        # TODO: Add training resumption
        model = self.make_model().to(self.device)
        with self.name_tracker.set_context("epoch", "init"):
            self.out_model.write(model=model, epoch="init")

        lr_init = self.lr_init
        lr_final = self.lr_final

        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

        total_iterations = self.n_epochs * len(dataloader)  # Match CMBNNCS's updates per batch, (not the more standard per epoch)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda iteration: (lr_final / lr_init) ** (iteration / total_iterations))

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            batch_n = 0
            for train_features, train_label, _ in tqdm(dataloader):
                batch_n += 1
                train_features = self.prep_data(train_features)
                train_label = self.prep_data(train_label)

                batch_loss = 0
                for _ in range(self.repeat_n):
                    optimizer.zero_grad()
                    output = model(train_features)
                    loss = loss_function(output, train_label)
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                scheduler.step()

                batch_loss = batch_loss / self.repeat_n

                if batch_n % self.output_every == 0:
                    logger.debug(f'Epoch {epoch+1}/{self.n_epochs}, Batch: {batch_n} complete. Batch loss: {batch_loss:.05f}')

                epoch_loss += batch_loss

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
        # For now, we consider only the most likely case that there's a single split
        #    being used for the creation of any PyTorch Dataset.
        # We cannot use the assets as I've created them, which use another object to
        #    track filenames.
        # The only thing that changes within a CMBMapDataset's filepaths is the sim #
        # So I create a single template string for each of the maps.
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = LabelledCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=NumpyMap()
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels, idx = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Data map resolution does not match configuration."
