import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
from omegaconf import DictConfig

import healpy as hp

from core import Split, Asset
from core.dataset import LabelledCMBMapDataset
from core.asset_handlers.handler_model_pytorch import PyTorchModel # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap
from .pytorch_model_base_executor import PetroffModelExecutor


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
        out_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        self.choose_device(cfg.model.petroff.train.device)

        self.lr = cfg.model.petroff.train.learning_rate
        self.n_epochs = cfg.model.petroff.train.n_epochs
        self.batch_size = cfg.model.petroff.train.batch_size
        self.output_every = cfg.model.petroff.train.output_every
        self.checkpoint = cfg.model.petroff.train.checkpoint_every
        self.extra_check = cfg.model.petroff.train.extra_check

        self.restart_epoch = cfg.model.petroff.train.restart_epoch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        # We use the first split as a template for setting up the dataset.
        #    This makes more sense in the context of inference over multiple
        #    different splits.
        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )

        self.inspect_data(dataloader)
        model = self.make_model()
        self.try_model(model)

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if self.restart_epoch is not None:
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            start_epoch = self.in_model.read(model=model, epoch=self.restart_epoch, optimizer=optimizer)
        else:
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
            start_epoch = 0

        for epoch in range(start_epoch, self.n_epochs):
            epoch_loss = 0.0
            batch_n = 0
            for train_features, train_label, _ in tqdm(dataloader):
                batch_n += 1
                train_features = self.prep_data(train_features)
                train_label = self.prep_data(train_label)

                optimizer.zero_grad()
                output = model(train_features)
                loss = loss_function(output, train_label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Output intermittently so progress is known
                if batch_n % self.output_every == 0:
                    logger.debug(f'Epoch {epoch+1}/{self.n_epochs}, Batch: {batch_n} complete. Loss: {loss.item()}')

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

        # dataset = LabelledCMBMapDataset(
        #     n_sims = template_split.n_sims,
        #     detectors = self.experiment.detector_freqs,
        #     label_path_template=cmb_path_template, 
        #     label_handler=HealpyMap(self.experiment),
        #     feature_path_template=obs_path_template,
        #     feature_handler=ManyHealpyMaps(self.experiment)
        #     )
        dataset = LabelledCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap()
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels, idx = next(iter(dataloader))
        logger.info(f"TrainingExecutor.preview_data() Feature batch shape: {train_features.size()}")
        logger.info(f"TrainingExecutor.preview_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Data map resolution does not match configuration."
