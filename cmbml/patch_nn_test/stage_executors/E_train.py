import logging

from tqdm import tqdm

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.asset_handlers_base import Config
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.handler_npymap import NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TrainCMBPatchDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.patch_nn_test.utils.display_help import show_patch


logger = logging.getLogger(__name__)


# TODO: Move elsewhere
class BasePyTorchModelExecutor(BaseStageExecutor):
    dtype_mapping = {
        "float": torch.float32,
        "double": torch.float64
    }

    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.n_dets = len(self.instrument.dets)
        self.nside = cfg.scenario.nside

    def choose_device(self, force_device=None) -> None:
        if force_device:
            self.device = force_device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    def make_fn_template(self, split: Split, asset):
        context = dict(
            split=split.name,
            sim=self.name_tracker.sim_name_template,
            freq="{freq}"
        )
        with self.name_tracker.set_contexts(contexts_dict=context):
            this_path_pattern = str(asset.path)
        return this_path_pattern

    def match_data_precision(self, tensor):
        # TODO: Revisit
        # data_precision is the precision with which the data is written to file
        # model_precision is the precision with which the model is created
        # tensor is the loaded data
        # If the tensor precision doesn't match the models, convert it
        # If the tensor precision doesn't match data_precision... is there an issue?
        if self.model_precision == "float" and tensor.dtype is torch.float64:
            return tensor.float()
        if self.model_precision == "float" and tensor.dtype is torch.float32:
            return tensor
        else:
            message = f"BasePyTorchModelExecutor data conversion is partially implemented. Received from config model precision: {self.model_precision}, data precision: {self.data_precision}. Received a tensor with dtype: {tensor.dtype}."
            logger.error(message)
            raise NotImplementedError(message)


class TrainingExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]
        # self.in_norm: Asset = self.assets_in["norm_file"]  # We may need this later
        in_model_handler: PyTorchModel
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        # in_norm_handler: Config

        self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.train.device)
        self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        self.batch_size = cfg.model.patch_nn.train.batch_size

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        template_split = self.splits[0]
        dataset = self.set_up_dataset(template_split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # TODO: Change to True (when using the full dataset)
            )

        # model = self.make_model().to(self.device)
        # loss_function = torch.nn.L1Loss(reduction='mean')
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        start_epoch = 0  # Temporary TODO: Remove, replace with epoch restart stuff from CMBNNCS

        for epoch in range(start_epoch, self.n_epochs):
            batch_n = 0
            with tqdm(dataloader, postfix={'Loss': 0}) as pbar:
                for train_features, train_label, sim_idx, p_idx in pbar:
                    logger.debug(f"train label shape: {train_label.shape}")
                    logger.debug(f"train features shape: {train_features.shape}")
                    for i in range(self.batch_size):
                        show_patch(train_label[i, :], train_features[i, :], 
                                   f"Batch {batch_n}, Sample {i}, Train{sim_idx[i]:04d}, Patch {p_idx[i]}")
                    if sim_idx[-1] >= 10 - self.batch_size:
                        # I have 10 sims, so this will show the last full batch (and crash after)
                        break
                    batch_n += 1
            break

    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        with self.name_tracker.set_context("split", template_split.name):
            which_patch_dict = self.get_patch_dict()

        dataset = TrainCMBPatchDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template,
            label_handler=self.in_cmb_asset.handler,
            # label_handler=HealpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=self.in_obs_assets.handler,
            which_patch_dict=which_patch_dict,
            nside_obs=self.nside,
            nside_patches=self.nside_patch
            # feature_handler=HealpyMap()
            )
        return dataset

    def get_patch_dict(self):
        patch_dict = self.in_all_p_ids_asset.read()
        patch_dict = patch_dict["patch_ids"]
        return patch_dict

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1] * train_features.size()[-2]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
