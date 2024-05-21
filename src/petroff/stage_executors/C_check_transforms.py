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
from petroff.preprocessing.scale_methods_factory import get_scale_class
from petroff.preprocessing.pytorch_transform_pixel_reorder import ReorderTransform
from core.pytorch_transform import TrainToTensor, train_remove_map_fields, train_add_map_fields


logger = logging.getLogger(__name__)


class CheckTransformsExecutor(PetroffModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="train")

        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_norm: Asset = self.assets_in["norm_file"]
        out_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config
        self.scale_class = None
        self.unscale_class = None
        self.set_scale_classes(cfg)

        model_precision = cfg.model.petroff.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]

    def set_scale_classes(self, cfg):
        scale_method = cfg.model.petroff.preprocess.scaling
        self.scale_class = get_scale_class(method=scale_method, 
                                           dataset="train", 
                                           scale="scale")
        self.unscale_class = get_scale_class(method=scale_method, 
                                             dataset="train", 
                                             scale="unscale")

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
            # No transforms for baseline
            pt_xforms=[],
            hp_xforms=[]
            )

        dataloader_raw = DataLoader(
            dataset_raw, 
            batch_size=1, 
            shuffle=False
            )

        obs_raw, cmb_raw = next(iter(dataloader_raw))

        # tensor_transform = TrainToTensor(dtype=self.dtype, device="cpu")
        dtype_transform = TrainToTensor(dtype=self.dtype)
        # tensor_only_transform = TrainToTensor()
        scale = self.scale_class(all_map_fields=self.map_fields,
                                 scale_factors=scale_factors,
                                 device="cpu",
                                 dtype=self.dtype)
        device_transform = TrainToTensor(device="cpu")
        pt_transforms = [
            dtype_transform,
            scale,
            train_remove_map_fields,
            device_transform
        ]

        reorder_transform_in = ReorderTransform(from_ring=True)
        hp_transforms = [
            reorder_transform_in
        ]

        dataset_prep = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            # Transform same as preprocessing to be done in the train loop
            pt_xforms=pt_transforms,
            hp_xforms=hp_transforms
            )

        dataloader_prep = DataLoader(
            dataset_prep, 
            batch_size=1, 
            shuffle=False
            )

        data = next(iter(dataloader_prep))

        obs_part, cmb_part = data

        # Inverse transforms as done during inference
        unscale = self.unscale_class(all_map_fields=self.map_fields,
                                                    scale_factors=scale_factors,
                                                    device="cpu",
                                                    dtype=self.dtype)
        pt_untransforms = [
            train_add_map_fields,  # Only needed for observations, no impact on cmb predictions (because of singleton dimensions in scale factors for CMB)
            unscale
            ]
        reorder_transform_out = ReorderTransform(from_ring=False)
        hp_untransforms = [
            reorder_transform_out
        ]

        for t in pt_untransforms:
            data = t(data)
        obs_post, cmb_post = data

        obs_post = obs_post.squeeze().numpy()
        cmb_post = cmb_post.squeeze().numpy()

        # For each observation frequency:
        all_temp = []
        for i in range(obs_post.shape[0]):
            temp = obs_post[i, :]
            # For each healpy untransform (which have to be applied to single maps)
            for t in hp_untransforms:
                temp = t(temp)
            all_temp.append(temp)
        obs_post = np.array(all_temp)

        # And for the cmb:
        for t in hp_untransforms:
            cmb_post = t(cmb_post)

        obs_raw = obs_raw.squeeze().numpy()
        cmb_raw = cmb_raw.squeeze().numpy()

        # Find the largest difference for each
        obs_delta = np.abs(obs_post - obs_raw)
        cmb_delta = np.abs(cmb_post - cmb_raw)

        logger.info(f"When trying the pre- and post- processing transforms: max observations delta is {obs_delta.max()}, obs_post mean:{obs_post.mean()}, obs_post_std:{obs_post.std()}, obs_raw mean: {obs_raw.mean()}, obs_raw std: {obs_raw.std()}")
        logger.info(f"When trying the pre- and post- processing transforms: max cmb delta is {cmb_delta.max()}")

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"TrainingExecutor.preview_data() Feature batch shape: {train_features.size()}")
        logger.info(f"TrainingExecutor.preview_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Data map resolution does not match configuration."
