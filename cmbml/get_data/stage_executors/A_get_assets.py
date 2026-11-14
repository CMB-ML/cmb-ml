import shutil
import json
import logging
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import HealpyMap, Mover, QTableHandler

from cmbml.get_data.utils.get_planck_data_ext import (
    get_planck_obs_data_ext, 
    get_planck_pred_data_ext, 
    get_planck_hm_data_ext
    )
from cmbml.get_data.utils.get_wmap_data_ext import get_wmap_chains_ext
from cmbml.get_data.utils.planck_box_download import download_planck_assets
from cmbml.get_data.utils.wmap_box_extract import extract_wmap_archive


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
        self.mask_nilc_map: Asset = self.assets_out.get('mask_nilc_map', None)

        # NOTE: The next two need getter functions!!!
        self.mask_point_source_map: Asset = self.assets_out.get('mask_point_source_map', None)
        self.mask_gal_cut_apo_0_map: Asset = self.assets_out.get('mask_gal_cut_apo_0_map', None)
        self.deltabandpass: Asset = self.assets_out['deltabandpass']
        # For reference:
        in_noise_varmaps: HealpyMap
        in_wmap_chains: Mover
        in_mask_map: HealpyMap
        in_delta_bandpass: QTableHandler

        self.detectors = list(cfg.scenario.detector_freqs)
        # self.wmap_chains_version = cfg.model.sim.cmb.wmap_chain_version

    def execute(self) -> None:
        """
        Download stuff.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        logger.info("Getting Planck assets from the public Box mirror.")
        self.get_planck_box_assets()

        logger.info("Getting WMAP9 chains from the public Box mirror.")
        self.get_wmap_chains()

        logger.info("Copying bundled CMB-ML assets.")
        self.copy_cmb_ml_assets()

    def get_planck_box_assets(self):
        # Resolve the downloaded-data directory through the existing asset paths.
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.noise_src_varmaps.path
        planck_dir = dummy_fp.parent

        # get_assets.py is run from a repository checkout; the manifest ships
        # with that checkout and does not need to be copied into CMB_ML_DATA.
        repo_root = Path(__file__).resolve().parents[3]
        manifest = repo_root / 'assets' / 'CMB-ML' / 'Planck_assets_manifest.json'
        download_planck_assets(manifest, planck_dir)

    def get_wmap_chains(self):
        # Use the configured external assets root, not the chain-version-specific
        # path (which may be absent from configurations that do not use chains).
        assets_root = self.deltabandpass.path.parent.parent
        wmap_dir = assets_root / 'WMAP'
        repo_root = Path(__file__).resolve().parents[3]
        manifest_path = repo_root / 'assets' / 'CMB-ML' / 'WMAP_assets_manifest.json'
        download_planck_assets(manifest_path, wmap_dir)

        # The simulation pipeline expects directories, not tar.gz filenames.
        for entry in json.loads(manifest_path.read_text())['files']:
            extract_wmap_archive(wmap_dir / entry['name'])

    def get_planck_obs_full_data(self):
        # Cheating a bit to use the name tracker; we don't have a filename but just need the parent directory
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.noise_src_varmaps.path
        noise_src_dir = dummy_fp.parent
        noise_src_dir.mkdir(parents=True, exist_ok=True)
        for det in tqdm(self.detectors):
            get_planck_obs_data_ext(detector=det, assets_directory=noise_src_dir, progress=True)  # download the data if it doesn't exist

    def get_planck_obs_hm_data(self):
        # Cheating a bit to use the name tracker; we don't have a filename but just need the parent directory
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.noise_src_varmaps.path
        noise_src_dir = dummy_fp.parent
        noise_src_dir.mkdir(parents=True, exist_ok=True)
        for det in tqdm(self.detectors):
            get_planck_hm_data_ext(detector=det, assets_directory=noise_src_dir, progress=True)  # download the data if it doesn't exist

    def get_planck_pred(self):
        if self.mask_nilc_map is None:
            return
        fp = self.mask_nilc_map.path
        fp.parent.mkdir(parents=True, exist_ok=True)
        get_planck_pred_data_ext(assets_directory=fp.parent, 
                                 fn=fp.name,
                                 file_size=self.mask_nilc_map.file_size,
                                 progress=True)  # download the data if it doesn't exist

    def copy_cmb_ml_assets(self):
        """Copy the entire bundled assets/CMB-ML directory into CMB_ML_DATA."""
        repo_root = Path(__file__).resolve().parents[3]
        src = repo_root / 'assets' / 'CMB-ML'
        dst = self.deltabandpass.path.parent
        if not src.is_dir():
            raise FileNotFoundError(f"Bundled CMB-ML assets not found: {src}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info("Copied CMB-ML assets: %s -> %s", src, dst)
