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
from .wrapper_api import SomeClass


logger = logging.getLogger(__name__)


class PredictionExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        logger.debug("Initializing SomeMethod Predict Executor")
        super().__init__(cfg, stage_str = "predict")

        # Define assets
        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_planck_deltabandpass: Asset = self.assets_in["planck_deltabandpass"]
        in_obs_handler: HealpyMap
        in_planck_deltabandpass_handler: QTableHandler

        # Pull some parameters from the config
        self.my_paramA = cfg.model.my_method.paramA
        self.my_paramB = cfg.model.my_method.paramB
        self.my_paramC = cfg.model.my_method.paramC

        # For example, the channels and fwhms are likely to be needed parameters
        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()
        self.fwhms = [det.fwhm for det in self.instrument.dets.values()]

        # We make a placeholder for the API object. We won't instantiate it 
        #    until later, as it may be expensive to do so.
        self.my_api = None

    def execute(self) -> None:
        """
        In this case, we simply iterate through all splits.
        """
        # Load things needed for the API object at the start of execute()
        mask = self.in_mask.read()

        # Instantiate the API object (just once, outside the loops)
        self.my_api = SomeClass(paramA=self.my_paramA, 
                                paramB=self.my_paramB, 
                                paramC=self.my_paramC,
                                mask=mask)

        # Iterate through all splits
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
        Now we use the API for each simulation.
        """
        # Load maps for this simulation
        input_maps = []
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                input_maps.append(self.in_obs_assets.read())

        # Assuming a very simple API...
        result = self.my_api.run(input_maps)

        self.out_cmb_asset.write(result)

        # It's likely that the API is more complicated. Loading things into this executor is A-ok!
