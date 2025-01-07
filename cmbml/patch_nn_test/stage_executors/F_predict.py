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
from cmbml.core.asset_handlers.asset_handlers_base import Config
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.handler_npymap import NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TestCMBPatchDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.patch_nn_test.dummy_model import SimpleUNetModel


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

        self.in_model_asset: Asset = self.assets_in["model"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_lut_asset: Asset = self.assets_in["lut"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_obs_map_handler: HealpyMap
        in_lut_handler: NumpyMap
        # in_norm_handler: Config

        self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.test.device)
        # self.n_epochs   = cfg.model.patch_nn.test.n_epochs
        self.batch_size = cfg.model.patch_nn.test.batch_size
        self.lut = self.in_lut_asset.read()
        self.dtype = self.dtype_mapping[cfg.model.patch_nn.dtype]

        self.model = None  # Placeholder for model

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

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

    def process_split(self, split):
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
        self.out_cmb_asset.write(data=pred_cmb)

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

        dataset = TestCMBPatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            sim_idx=0,
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
