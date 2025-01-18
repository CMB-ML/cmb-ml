import logging

from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.core.asset_handlers import (
    Config, 
    EmptyHandler,
    HealpyMap,
    QTableHandler 
    )
from cmbml.utils import make_instrument, Instrument
from cmbml.utils.suppress_print import SuppressPrint
from .make_config import ConfigMaker
from .wrapper_api import run_wrapped_api
from .wrapper_partial_api import run_partial_api
from .wrapper_cfg import run_configed_script 


logger = logging.getLogger(__name__)


class PredictionExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        logger.debug("Initializing NILC Predict Executor")
        super().__init__(cfg, stage_str = "predict")

        self.out_config: Asset = self.assets_out["config_file"]
        self.out_model: Asset = self.assets_out["model"]         # For the working directory
        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_config_handler: Config
        out_model_handler: EmptyHandler
        out_cmb_map_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_deltabandpass: Asset = self.assets_in["deltabandpass"]
        in_obs_handler: HealpyMap
        in_deltabandpass_handler: QTableHandler

        in_det_table: Asset = self.assets_in['deltabandpass']

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()

        self.model_cfg_maker = ConfigMaker(cfg, det_info)

    def execute(self) -> None:
        """
        In this case, we simply iterate through all splits.
        """
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, 
                      split: Split) -> None:
        """
        In this case, we simply iterate through all simulations in a split.
        """
        logger.info(f"Processing split: {split.name}, for {split.n_sims} simulations.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        """
        To run some code, we set up the working directory, run the code, and then move the result.
        """
        # Set up the working directory
        working_path = self.out_model.path
        working_path.mkdir(exist_ok=True, parents=True)

        # Get filenames for the input maps; these need to go into the config file
        input_paths = []
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                path = self.in_obs_assets.path
                # Convert to string; we're going to convert this information to a yaml file
                input_paths.append(str(path))

        # Get the filename for the mask; this also needs to go into the config file
        if self.in_mask is not None:
            mask_path = self.in_mask.path
        else:
            mask_path = None
        
        # Make the config file
        cfg_dict = self.model_cfg_maker.make_config(output_path=working_path,
                                                    input_paths=input_paths,
                                                    mask_path=mask_path)

        # Write the config file
        self.out_config.write(data=cfg_dict, verbose=False)

        # Non-functioning examples of different kinds of methods that work...
        method = None
        if method == "wrapped_api":
            # SuppressPrint keeps the console clear. It is useful for long-running code.
            with SuppressPrint():
                # logger.debug("Running PyILC Code...")
                run_wrapped_api(self.out_config.path)
        elif method == "partial_api":
            run_partial_api(self.out_config.path)
        elif method == "configed_script":
            run_configed_script(self.out_config.path)
        else:
            raise ValueError(f"Method {method} not recognized.")

        self.move_result()
        self.clear_pyilc_working_directory()

    def run_script(self, script_path: str) -> None:
        """
        Outside scripts may be used. We can run them here.
        """
        logger.debug(f"Running script: {script_path}")

    def move_result(self):
        """
        Outside scripts may save to a hardcoded file name. We can
        move the result to the final destination.
        """
        logger.debug("Moving result to final destination.")
        result_dir = self.out_model.path
        result_prefix = self.cfg.model.pyilc.output_prefix
        result_ext = self.cfg.model.pyilc.save_as
        result_fn = f"{result_prefix}needletILCmap_component_CMB.{result_ext}"

        result_path = result_dir / result_fn
        destination_path = self.out_cmb_asset.path

        destination_path.parent.mkdir(exist_ok=True, parents=True)

        result_path.rename(destination_path)

    def clear_pyilc_working_directory(self):
        """
        Outside scripts may use a working directory. That can be cleared between runs.
        """
        logger.debug("Clearing working directory.")
        working_path = self.out_model.path
        for file in working_path.iterdir():
            file.unlink()
        return
