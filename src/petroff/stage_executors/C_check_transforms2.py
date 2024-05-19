from typing import List, Callable

import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
from omegaconf import DictConfig

import numpy as np

import healpy as hp

from core import Split, Asset
from core.asset_handlers.asset_handlers_base import Config
from core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap
from .pytorch_model_base_executor import PetroffModelExecutor
from core.pytorch_dataset import TrainCMBMapDataset
from petroff.pytorch_transform_absmax_scale import TrainAbsMaxScaleMap
from petroff.pytorch_transform_absmax_scale import TrainAbsMaxUnScaleMap
from petroff.pytorch_transform_pixel_reorder import ReorderTransform
from core.pytorch_transform import TrainToTensor


logger = logging.getLogger(__name__)


class CheckTransformsExecutor(PetroffModelExecutor):
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
        # self.choose_device(cfg.model.petroff.train.device)

        # self.lr = cfg.model.petroff.train.learning_rate
        # self.n_epochs = cfg.model.petroff.train.n_epochs
        # self.batch_size = cfg.model.petroff.train.batch_size
        # self.output_every = cfg.model.petroff.train.output_every
        # self.checkpoint = cfg.model.petroff.train.checkpoint_every
        # self.extra_check = cfg.model.petroff.train.extra_check
        # self.num_workers = cfg.model.petroff.train.get("num_workers", 0)

        self.restart_epoch = cfg.model.petroff.train.restart_epoch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        template_split = self.splits[0]
        scale_factors = self.in_norm.read()

        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset_raw = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            transforms=[],
            hp_transforms=[]
            )

        dataloader_raw = DataLoader(
            dataset_raw, 
            batch_size=1, 
            shuffle=False
            )

        obs_raw, cmb_raw = next(iter(dataloader_raw))

        dtype_transform = TrainToTensor(self.dtype, device="cpu")

        scale = TrainAbsMaxScaleMap(all_map_fields=self.map_fields,
                                    scale_factors=scale_factors,
                                    device="cpu",
                                    dtype=self.dtype)
        reorder_transform_in = ReorderTransform(from_ring=True)

        dataset_prep = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            transforms=[dtype_transform, scale],
            hp_transforms=[reorder_transform_in]
            # hp_transforms=[]
            )

        dataloader_prep = DataLoader(
            dataset_prep, 
            batch_size=1, 
            shuffle=False
            )

        data = next(iter(dataloader_prep))

        untransform_stack = [
            TrainAbsMaxUnScaleMap(all_map_fields=self.map_fields,
                                                scale_factors=scale_factors,
                                                device="cpu",
                                                dtype=self.dtype)
            ]

        # Use a list comprehension to apply a series of transforms; unlist with [0]
        data = [t(data) for t in untransform_stack][0]
        obs_post, cmb_post = data

        reorder_transform_out = ReorderTransform(from_ring=False)

        obs_post = obs_post.squeeze().numpy()
        cmb_post = cmb_post.squeeze().numpy()

        obs_post = np.array([
            reorder_transform_out(obs_post[i, :]) for i in range(obs_post.shape[0])
        ])
        cmb_post = reorder_transform_out(cmb_post)

        obs_delta = np.abs(obs_post - obs_raw.numpy())
        cmb_delta = np.abs(cmb_post - cmb_raw.numpy())

        logger.info(f"When trying the pre- and post- processing transforms: observations delta is {obs_delta.max()}")
        logger.info(f"When trying the pre- and post- processing transforms: cmb delta is {cmb_delta.max()}")

    def set_up_dataset(self, template_split: Split, scale_factors) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dtype_transform = TrainToTensor(self.dtype, device="cpu")

        scale_map_transform = TrainAbsMaxScaleMap(all_map_fields=self.map_fields,
                                                  scale_factors=scale_factors,
                                                  device="cpu",
                                                  dtype=self.dtype)

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            transforms=[dtype_transform, scale_map_transform]
            )
        return dataset

    def set_up_no_xform_dataset(self, template_split:Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            transforms=[]
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"TrainingExecutor.preview_data() Feature batch shape: {train_features.size()}")
        logger.info(f"TrainingExecutor.preview_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Data map resolution does not match configuration."
