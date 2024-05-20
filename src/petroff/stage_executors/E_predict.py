import logging

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from omegaconf import DictConfig

from core import Split, Asset
from core.pytorch_dataset import TestCMBMapDataset
from core.asset_handlers.asset_handlers_base import Config
from core.asset_handlers.pytorch_model_handler import PyTorchModel
from .pytorch_model_base_executor import PetroffModelExecutor
from core.asset_handlers.healpy_map_handler import HealpyMap
from petroff.scaling.scale_methods_factory import get_scale_class
from core.pytorch_transform import TestToTensor
from petroff.pytorch_transform_pixel_reorder import ReorderTransform


logger = logging.getLogger(__name__)


class PredictionExecutor(PetroffModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_norm: Asset = self.assets_in["norm_file"]
        self.in_model: Asset = self.assets_in["model"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        in_norm_file_handler: Config
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel

        model_precision = cfg.model.petroff.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.petroff.predict.device)
        self.batch_size = cfg.model.petroff.predict.batch_size
        self.scale_class = None
        self.unscale_class = None
        self.set_scale_classes(cfg)

        self.postprocess = None
        self.hp_postprocesses = None

    def set_scale_classes(self, cfg):
        scale_method = cfg.model.petroff.prep.scaling
        self.scale_class = get_scale_class(method=scale_method, 
                                           dataset="test", 
                                           scale="scale")
        self.unscale_class = get_scale_class(method=scale_method, 
                                             dataset="test", 
                                             scale="unscale")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        for model_epoch in self.model_epochs:
            logger.info(f"Making predictions based on epoch {model_epoch}")
            model = self.make_model()
            with self.name_tracker.set_context("epoch", model_epoch):
                self.in_model.read(model=model, epoch=model_epoch)
            model.eval().to(self.device)
            for split in self.splits:
                context = dict(split=split.name, epoch=model_epoch)
                with self.name_tracker.set_contexts(contexts_dict=context):
                    self.process_split(model, split)

    def process_split(self, model, split):
        dataset = self.set_up_dataset(split)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
            )

        with torch.no_grad():
            for features, idcs in tqdm(dataloader):
                predictions = model(features)
                for pred, idx in zip(predictions, idcs):
                    with self.name_tracker.set_context("sim_num", idx.item()):
                        pred_npy = self.post_process(pred)
                        # print(pred_npy)[..., :10]
                        # print(pred)[..., :10]
                        self.out_cmb_asset.write(data=pred_npy)

    def post_process(self, pred):
        unscale_transform = self.postprocess_unscale
        pred = pred.detach().cpu()
        pred = unscale_transform(pred)
        pred = pred.numpy()
        for post in self.hp_postprocesses:
            pred = post(pred)
        return pred

    def set_up_dataset(self, template_split: Split) -> None:
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)
        scale_factors = self.in_norm.read()

        dtype_transform = TestToTensor(self.dtype, device="cpu")
        scale_map_transform = self.scale_class(all_map_fields=self.map_fields,
                                               scale_factors=scale_factors,
                                               device="cpu",
                                               dtype=self.dtype)
        device_transform = TestToTensor(self.dtype, device=self.device)

        pt_transforms = [
            dtype_transform, 
            scale_map_transform, 
            device_transform
            ]

        hp_transforms = [
            ReorderTransform(from_ring=False)
            ]

        dataset = TestCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            feature_path_template=obs_path_template,
            file_handler=HealpyMap(),
            transforms=pt_transforms,
            hp_xforms=hp_transforms
            )

        self.postprocess_unscale = self.unscale_class(all_map_fields=self.map_fields,
                                                      scale_factors=scale_factors,
                                                      device="cpu",
                                                      dtype=self.dtype)
        self.hp_postprocesses = [ReorderTransform(from_ring=False)]
        return dataset
