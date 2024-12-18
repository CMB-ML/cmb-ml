import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler

from get_data.utils.get_planck_data_ext import get_planck_obs_data_ext
from get_data.utils.get_wmap_data_ext import get_wmap_chains_ext


logger = logging.getLogger(__name__)


class GetAssetsExecutor(BaseStageExecutor):
    """
    GetAssetsExecutor downloads assets needed for running CMB-ML.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='raw')

        self.noise_src_varmaps: Asset = self.assets_out['noise_src_varmaps']
        self.wmap_chains: Asset = self.assets_out['wmap_chains']
        # For reference:
        in_noise_sim: HealpyMap

        self.detectors = list(cfg.scenario.full_instrument.keys())

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        logger.info("Getting Planck observations.")
        self.get_noise_src_varmaps()
        logger.info("Getting WMAP chains.")
        self.get_wmap_chains()

    def get_wmap_chains(self):
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.wmap_chains.path
        wmap_dir = dummy_fp.parent
        wmap_dir.mkdir(parents=True, exist_ok=True)
        get_wmap_chains_ext(assets_directory=wmap_dir, mnu=True, progress=True)

    def get_noise_src_varmaps(self):
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.noise_src_varmaps.path
        noise_src_dir = dummy_fp.parent
        noise_src_dir.mkdir(parents=True, exist_ok=True)
        for det in tqdm(self.detectors):
            get_planck_obs_data_ext(detector=det, assets_directory=noise_src_dir, progress=True)  # This will download the data if it doesn't exist
