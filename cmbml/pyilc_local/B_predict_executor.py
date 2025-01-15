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
from .make_pyilc_config import ILCConfigMaker
from cmbml.pyilc_redir.pyilc_wrapper import run_ilc
from cmbml.utils.suppress_print import SuppressPrint


logger = logging.getLogger(__name__)


class PredictionExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        logger.debug("Initializing NILC PredictExecutor")
        super().__init__(cfg, stage_str = "predict")

        self.out_config: Asset = self.assets_out["config_file"]
        self.out_model: Asset = self.assets_out["model"]
        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_config_handler: Config
        out_model_handler: EmptyHandler
        out_cmb_map_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_planck_deltabandpass: Asset = self.assets_in["planck_deltabandpass"]
        in_obs_handler: HealpyMap
        in_planck_deltabandpass_handler: QTableHandler

        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        # with self.name_tracker.set_context("src_root", cfg.local_system.assets_dir):
        #     planck_bandpass = self.in_planck_deltabandpass.read()
        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()

        self.model_cfg_maker = ILCConfigMaker(cfg, det_info)

    def execute(self) -> None:
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Executing PredictExecutor process_split() for split: {split.name}, for {split.n_sims} simulations.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        working_path = self.out_model.path
        working_path.mkdir(exist_ok=True, parents=True)

        input_paths = []
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                path = self.in_obs_assets.path
                # Convert to string; we're going to convert this information to a yaml file
                input_paths.append(str(path))

        if self.in_mask is not None:
            mask_path = self.in_mask.path
        else:
            mask_path = None
        cfg_dict = self.model_cfg_maker.make_config(output_path=working_path,
                                                    input_paths=input_paths,
                                                    mask_path=mask_path)
        self.out_config.write(data=cfg_dict, verbose=False)
        # logger.debug("Running PyILC Code...")
        with SuppressPrint():
            run_ilc(self.out_config.path)
        # logger.debug("Moving resulting map.")
        self.move_result()
        self.clear_pyilc_working_directory()

    def move_result(self):
        result_dir = self.out_model.path
        result_prefix = self.cfg.model.pyilc.output_prefix
        result_ext = self.cfg.model.pyilc.save_as
        result_fn = f"{result_prefix}needletILCmap_component_CMB.{result_ext}"

        result_path = result_dir / result_fn
        destination_path = self.out_cmb_asset.path

        destination_path.parent.mkdir(exist_ok=True, parents=True)

        result_path.rename(destination_path)

    def clear_pyilc_working_directory(self):
        working_path = self.out_model.path
        for file in working_path.iterdir():
            file.unlink()
        return