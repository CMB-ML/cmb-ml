import logging

from tqdm import tqdm

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseCMBNNCSModelExecutor
from core import Split, Asset
from core.asset_handlers.asset_handlers_base import Config
from core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
# from core.asset_handlers.healpy_map_handler import HealpyMap
from cmbnncs_local.handler_npymap import NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbnncs_local.dataset import TrainCMBMapDataset
from core.pytorch_transform import TrainToTensor
from cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
from cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect


logger = logging.getLogger(__name__)


class TrainingExecutor(BaseCMBNNCSModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_norm: Asset = self.assets_in["norm_file"]
        in_model_handler: PyTorchModel
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_norm_handler: Config

        self.norm_data = None

        model_precision = cfg.model.cmbnncs.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.cmbnncs.train.device)

        self.lr_init = cfg.model.cmbnncs.train.learning_rate
        self.lr_final = cfg.model.cmbnncs.train.learning_rate_min
        self.repeat_n = cfg.model.cmbnncs.train.repeat_n
        self.n_epochs = cfg.model.cmbnncs.train.n_epochs
        self.batch_size = cfg.model.cmbnncs.train.batch_size
        self.checkpoint = cfg.model.cmbnncs.train.checkpoint_every
        self.extra_check = cfg.model.cmbnncs.train.extra_check
        self.scale_class = None
        self.set_scale_class(cfg)

        self.restart_epoch = cfg.model.cmbnncs.train.restart_epoch

    def set_scale_class(self, cfg):
        scale_method = cfg.model.cmbnncs.preprocess.scaling
        self.scale_class = get_scale_class(method=scale_method, 
                                           dataset="train", 
                                           scale="scale")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        dets_str = ', '.join([str(k) for k in self.instrument.dets.keys()])
        logger.info(f"Creating model using detectors: {dets_str}")

        logger.info(f"Using exponential learning rate scheduler.")
        logger.info(f"Initial learning rate is {self.lr_init}")
        logger.info(f"Final minimum learning rate is {self.lr_final}")
        logger.info(f"Number of epochs is {self.n_epochs}")
        logger.info(f"Batch size is {self.batch_size}")
        logger.info(f"Checkpoint every {self.checkpoint} iterations")
        logger.info(f"Extra check is set to {self.extra_check}")

        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        # mp.set_start_method('spawn')
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            # num_workers=1
            )
        # self.inspect_data(dataloader)

        model = self.make_model().to(self.device)
        # self.try_model(model)

        lr_init = self.lr_init
        lr_final = self.lr_final
        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

        total_iterations = self.n_epochs * len(dataloader)  # Match CMBNNCS's updates per batch, (not the more standard per epoch)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda iteration: (lr_final / lr_init) ** (iteration / total_iterations))

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer, 
                                                 scheduler=scheduler)
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
            batch_loss = 0
            with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
                for train_features, train_label in pbar:
                    batch_n += 1

                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    for _ in range(self.repeat_n):
                        optimizer.zero_grad()
                        output = model(train_features)
                        loss = loss_function(output, train_label)
                        loss.backward()
                        optimizer.step()
                        batch_loss += loss.item()
                    scheduler.step()
                    pbar.set_postfix({'Loss': loss.item() / self.batch_size})

                    batch_loss = batch_loss / self.repeat_n

                    epoch_loss += batch_loss

            epoch_loss /= len(dataloader.dataset)
            
            logger.info(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}')

            # Checkpoint every so many epochs
            if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         epoch=epoch + 1,
                                         loss=epoch_loss)


    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        # scale_factors = self.in_norm.read()
        # dtype_transform = TrainToTensor(self.dtype, device="cpu")
        # scale_map_transform = self.scale_class(all_map_fields=self.map_fields,
        #                                        scale_factors=scale_factors,
        #                                        device="cpu",
        #                                        dtype=self.dtype)
        # device_transform = TrainToTensor(self.dtype, device=self.device)
        # pt_transforms = [
        #     dtype_transform,
        #     scale_map_transform,
        #     device_transform
        # ]

        # reorder_transform_in = sphere2rect
        # np_transforms = [
        #     reorder_transform_in
        # ]

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            label_handler=NumpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=NumpyMap(),
            # pt_xforms=pt_transforms,
            # hp_xforms=np_transforms
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1] * train_features.size()[-2]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
