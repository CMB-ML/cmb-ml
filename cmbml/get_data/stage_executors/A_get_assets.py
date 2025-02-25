import shutil
import logging
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import HealpyMap, Mover, QTableHandler

from cmbml.get_data.utils.get_planck_data_ext import get_planck_obs_data_ext, get_planck_pred_data_ext
from cmbml.get_data.utils.get_wmap_data_ext import get_wmap_chains_ext


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
        self.mask_src_map: Asset = self.assets_out.get('mask_src_map', None)
        self.deltabandpass: Asset = self.assets_out['deltabandpass']
        # For reference:
        in_noise_varmaps: HealpyMap
        in_wmap_chains: Mover
        in_mask_map: HealpyMap
        in_delta_bandpass: QTableHandler

        self.detectors = list(cfg.scenario.full_instrument.keys())

    def execute(self) -> None:
        """
        Download stuff.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")

        # Optional Planck observations are needed if you will be producing simulations (noise maps)
        logger.info("Getting Planck observations.")
        self.get_noise_src_varmaps()
        # Optional WMAP chains only needed if you will not be producing simulations (CMB maps)
        logger.info("Getting WMAP chains.")
        self.get_wmap_chains()

        # Needed for analysis
        logger.info("Getting Planck predicted data for the NILC mask.")  # TODO: Parameterize this
        self.get_src_maskmap()

        # CMB-ML assets include the detector information (needed for all stages) and the file
        #    information for the simulations (optional, but small, so it's lumped together)
        logger.info("Getting CMB-ML assets.")
        self.copy_cmb_ml_assets()

    def get_wmap_chains(self):
        # Cheating a bit to use the name tracker; we don't have a filename but just need the parent directory
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.wmap_chains.path
        wmap_dir = dummy_fp.parent
        wmap_dir.mkdir(parents=True, exist_ok=True)
        get_wmap_chains_ext(assets_directory=wmap_dir, mnu=True, progress=True)

    def get_noise_src_varmaps(self):
        # Cheating a bit to use the name tracker; we don't have a filename but just need the parent directory
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            dummy_fp = self.noise_src_varmaps.path
        noise_src_dir = dummy_fp.parent
        noise_src_dir.mkdir(parents=True, exist_ok=True)
        for det in tqdm(self.detectors):
            get_planck_obs_data_ext(detector=det, assets_directory=noise_src_dir, progress=True)  # download the data if it doesn't exist

    def get_src_maskmap(self):
        if self.mask_src_map is None:
            return
        fp = self.mask_src_map.path
        fp.parent.mkdir(parents=True, exist_ok=True)
        get_planck_pred_data_ext(assets_directory=fp.parent, 
                                 fn=fp.name,
                                 file_size=self.mask_src_map.file_size,
                                 progress=True)  # download the data if it doesn't exist

    def copy_cmb_ml_assets(self):
        """
        Copies the 'delta_bandpasses' folder from the repository assets to the user's asset directory.
        """
        # Define source and destination paths
        src = Path("./assets/CMB-ML").resolve()
        dst = self.deltabandpass.path.parent.resolve()

        try:
            # Inform the user about the copy operation
            print(f"Copying:\n  Source: {src}\n  Destination: {dst}")

            # Perform the copy operation
            shutil.copytree(src, dst, dirs_exist_ok=True)  # Allows overwriting existing directories
            print("Copy operation completed successfully.")
        except FileExistsError:
            print(f"Error: Destination folder already exists: {dst}")
        except FileNotFoundError:
            print(f"Error: Source folder does not exist: {src}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")