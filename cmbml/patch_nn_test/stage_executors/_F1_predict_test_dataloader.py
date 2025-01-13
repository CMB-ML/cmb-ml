import logging

from tqdm import tqdm

# import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TestCMBPatchDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.patch_nn_test.utils.display_help import show_patch
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class PredictTryDataloaderExecutor(BasePyTorchModelExecutor):
    """
    Goal: Reassemble a map from patches.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_handler: NumpyMap

        self.in_model_asset: Asset = self.assets_in["model"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_obs_map_handler: HealpyMap
        in_lut_handler: NumpyMap
        # in_norm_handler: Config

        self.nside_patch = cfg.model.patches.nside_patch

        # self.choose_device(cfg.model.patch_nn.test.device)
        # self.n_epochs   = cfg.model.patch_nn.test.n_epochs
        self.batch_size = cfg.model.patch_nn.test.batch_size
        self.lut = self.in_lut_asset.read()

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            with self.name_tracker.set_contexts(contexts_dict={"split": split.name}):
                self.process_split(split)

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

        # get target map from the dataset (which internally loads the full map; to be used for debugging but not in production)
        target_map = dataset._current_map_data[use_detector]

        assert np.all(reassembled_map == target_map), "Reassembled map does not match target map."
        logger.info("Success! Reassembled map matches target map!")

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

        dataset = TestCMBPatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            lut=self.lut,
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1] * train_features.size()[-2]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
