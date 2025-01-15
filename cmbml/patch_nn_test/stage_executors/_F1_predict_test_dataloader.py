import logging

from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
from cmbml.patch_nn_test.dataset import TestCMBPatchDataset
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.patch_nn_test.utils.minmax_scale import minmax_unscale, MinMaxScaler


logger = logging.getLogger(__name__)


class PredictTryDataloaderExecutor(BasePyTorchModelExecutor):
    """
    Goal: Reassemble a map from patches.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        # self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        # out_cmb_handler: NumpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        # self.in_model_asset: Asset = self.assets_in["model"]
        in_obs_map_handler: HealpyMap
        in_lut_handler: NumpyMap
        in_dataset_stats_handler: Config
        # in_model_handler: PyTorchModel

        self.scaling = cfg.model.patch_nn.get("scaling", None)
        if self.scaling and self.scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {self.scaling}."
            raise NotImplementedError(msg)

        self.batch_size = cfg.model.patch_nn.test.batch_size
        self.lut = self.in_lut_asset.read()
        self.dataset_stats = None  # Placeholder for dataset_stats (min/max values for normalization)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.load_dataset_stats()

        for split in self.splits:
            with self.name_tracker.set_contexts(contexts_dict={"split": split.name}):
                self.process_split(split)

    def load_dataset_stats(self) -> None:
        # TODO: Use a class to better handle scaling/normalization
        if self.scaling == "minmax":
            self.dataset_stats = self.in_dataset_stats.read()

    def process_split(self, split):
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        for sim_num in tqdm(split.iter_sims()):
            dataset.sim_idx = sim_num
            with self.name_tracker.set_context("sim_num", sim_num):
                self.process_sim(sim_num, dataloader, dataset)
            # Do just the first simulation for this test
            break

    def process_sim(self, sim_idx, dataloader, dataset):
        """
        In this test executor, the goal is to reconstruct just one of the input feature maps.
        """
        logger.info("Starting map")

        use_detector = 0  # Use 30 GHz (first frequency) for now

        this_map_results = []
        for test_features, verif_sim_idx, p_idx in dataloader:
            logger.info(f"train features shape: {test_features.shape}, input_sim_idx:{sim_idx}, verified sim_idx: {verif_sim_idx}, p_idx: {p_idx}")
            this_map_results.append(test_features[:, use_detector, ...])

        reassembled_map = self.reassemble_map(this_map_results)
        reassembled_map = self.unscale(reassembled_map, self.dataset_stats[30])

        # get target map from the dataset (which internally loads the full map; to be used for debugging but not in production)
        target_map = dataset._current_map_data[use_detector]

        assert np.allclose(reassembled_map, target_map), "Reassembled map does not match target map."
        # assert np.all(reassembled_map == target_map), "Reassembled map does not match target map."
        logger.info("Success! Reassembled map matches target map!")

    def unscale(self, map_array, cmb_dataset_stats):
        """
        Unscale the map_array using the dataset_stats.
        """
        new_map_array = np.zeros_like(map_array)
        # TODO: Implement multiple fields
        vmin = cmb_dataset_stats['I']['vmin'].value
        vmax = cmb_dataset_stats['I']['vmax'].value
        new_map_array = minmax_unscale(map_array, vmin, vmax)
        return new_map_array

    def reassemble_map(self, sim_results_list):
        sim_results = [sr.numpy() for sr in sim_results_list]
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
