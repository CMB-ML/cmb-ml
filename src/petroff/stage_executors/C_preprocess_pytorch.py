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
from src.core import Split, Asset
from src.core.asset_handlers.asset_handlers_base import Config
from src.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from src.core.asset_handlers.healpy_map_handler import HealpyMap
from src.core.pytorch_dataset import TrainCMBMapDataset
from src.core.pytorch_transform import TrainToTensor, train_remove_map_fields
from src.petroff.preprocessing.scale_methods_factory import get_scale_class
from src.petroff.preprocessing.pytorch_transform_pixel_reorder import ReorderTransform


logger = logging.getLogger(__name__)


class PreprocessExecutor(PetroffModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess")

        self.out_cmb: Asset = self.assets_out["cmb_map"]
        self.out_obs: Asset = self.assets_out["obs_maps"]
        out_cmb_map_handler: HealpyMap
        out_obs_map_handler: HealpyMap

        self.in_cmb: Asset = self.assets_in["cmb_map"]
        self.in_obs: Asset = self.assets_in["obs_maps"]
        self.in_norm: Asset = self.assets_in["norm_file"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_handler: Config

        self.norm_data = None

        model_precision = cfg.model.petroff.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]

        self.batch_size = cfg.model.petroff.preprocess.batch_size
        self.scale_class = None
        self.set_scale_class(cfg)
        self.num_workers = cfg.model.petroff.train.get("num_workers", 0)

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
        logger.info(f"Preprocessing for detectors: {dets_str}")

        logger.info(f"Using fixed learning rate.")
        logger.info(f"batch size is {self.batch_size}")

        for split in self.splits:
            self.process_split(split)

    def process_split(self, split):
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
            )

        with torch.no_grad():
            with tqdm(dataloader) as pbar:
                for i, (features, label) in enumerate(pbar):
                    context_params = dict(split=split.name, sim_num=i)
                    with self.name_tracker.set_contexts(contexts_dict=context_params):
                        self.process_batch(features, label)

    def process_batch(self, features, label):
        label = label.squeeze().numpy()
        self.out_cmb.write(data=label, nest=True)
        for i, freq in enumerate(self.instrument.dets):
            with self.name_tracker.set_context("freq", freq):
                # Remove the batch dim, get the i^th map, and convert to numpy
                obs_map = features.squeeze()[i].numpy()
                self.out_obs.write(data=obs_map, nest=True)

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb)
        obs_path_template = self.make_fn_template(template_split, self.in_obs)

        scale_factors = self.in_norm.read()
        dtype_transform = TrainToTensor(self.dtype, device="cpu")
        scale_map_transform = self.scale_class(all_map_fields=self.map_fields,
                                               scale_factors=scale_factors,
                                               device="cpu",
                                               dtype=self.dtype)
        remove_map_field_transform = train_remove_map_fields
        pt_transforms = [
            dtype_transform,
            scale_map_transform,
            remove_map_field_transform
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
