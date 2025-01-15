import logging

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
from cmbml.patch_nn_test.dataset import TestCMBPatchDataset
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.patch_nn_test.dummy_model import SimpleUNetModel
from cmbml.patch_nn_test.utils.minmax_scale import minmax_unscale, MinMaxScaler


logger = logging.getLogger(__name__)


class PredictExectutor(BasePyTorchModelExecutor):
    """
    Goal: Reassemble a map from patches.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_model_asset: Asset = self.assets_in["model"]
        in_obs_map_handler: HealpyMap
        in_lut_handler: NumpyMap
        in_dataset_stats_handler: Config
        in_model_handler: PyTorchModel

        self.scaling = cfg.model.patch_nn.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)

        self.choose_device(cfg.model.patch_nn.test.device)
        self.batch_size = cfg.model.patch_nn.test.batch_size
        self.lut = None
        self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]

        self.model = None  # Placeholder for model
        self.dataset_stats = None  # Placeholder for dataset_stats (min/max values for normalization)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.load_lut()
        self.load_dataset_stats()

        # It would likely be safer to have this within the loop, right before read()
        #    But this should work and be faster (especially with larger models)
        self.model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network")
        self.model.eval().to(self.device)

        with torch.no_grad():  # We don't need gradients for prediction
            for model_epoch in self.model_epochs:
                for split in self.splits:
                    context_dict = dict(split=split.name, epoch=model_epoch)
                    with self.name_tracker.set_contexts(context_dict):
                        self.in_model_asset.read(model=self.model, epoch=model_epoch)
                        self.process_split(split)

    def load_lut(self) -> None:
        self.lut = self.in_lut_asset.read()

    def load_dataset_stats(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()

    def process_split(self, split) -> None:
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            )
        for sim_num in tqdm(split.iter_sims()):
            dataset.sim_idx = sim_num
            with self.name_tracker.set_context("sim_num", sim_num):
                self.process_sim(sim_num, dataloader, dataset)

    def process_sim(self, sim_idx, dataloader, dataset):
        """
        Process the simulation using the trained model on each patch.
        """
        this_map_results = []
        for test_features, _, _ in dataloader:
            test_features = test_features.to(device=self.device, dtype=self.dtype)
            predictions = self.model(test_features)
            this_map_results.append(predictions)
        pred_cmb = self.reassemble_map(this_map_results)
        if self.scaling == "minmax":
            pred_cmb = self.unscale(pred_cmb, self.dataset_stats['cmb'])
        self.out_cmb_asset.write(data=pred_cmb)

    def unscale(self, map_array, dataset_stats):
        """
        Unscale the map_array using the dataset_stats.
        """
        new_map_array = np.zeros_like(map_array)
        # TODO: Implement multiple fields
        vmin = dataset_stats['I']['vmin']
        vmax = dataset_stats['I']['vmax']
        new_map_array = minmax_unscale(map_array, vmin, vmax)
        return new_map_array

    def reassemble_map(self, sim_results_list):
        sim_results = [sr.cpu().numpy() for sr in sim_results_list]
        # this_map_results now contains all patches for one frequency for one simulation
        #   as a list length n_p_id / batch_size of arrays with 
        #   shape (batch_size, patch_side, patch_side)
        # Comments will assume batch_size is 4, and we have 192 patches, each 128 x 128 pixels
        this_map_array = np.stack(sim_results, axis=0)  # Convert to array shape (48,4,128,128)
        this_map_array = this_map_array.reshape(-1, this_map_array.shape[-2], this_map_array.shape[-1])  # Convert to array shape (192,128,128)
        # this_map_array now contains all patches for one frequency for one simulation

        reassembled_map = np.zeros(np.prod(self.lut.shape), dtype=this_map_array.dtype)
        # Use the lut to reassemble the map
        reassembled_map[self.lut] = this_map_array
        return reassembled_map

    def set_up_dataset(self, template_split: Split) -> TestCMBPatchDataset:
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        transform = None
        if self.scaling == "minmax":
            vmins = np.array([self.dataset_stats[f]["I"]["vmin"].value for f in self.instrument.dets.keys()])
            vmaxs = np.array([self.dataset_stats[f]["I"]["vmax"].value for f in self.instrument.dets.keys()])
            transform = MinMaxScaler(vmins=vmins, vmaxs=vmaxs)

        dataset = TestCMBPatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            lut=self.lut,
            transform=transform
            )
        return dataset
