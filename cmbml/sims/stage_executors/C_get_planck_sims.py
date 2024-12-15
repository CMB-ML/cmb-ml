import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from get_data.utils.get_planck_data_ext import get_planck_noise_data_ext

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler


logger = logging.getLogger(__name__)


class GetPlanckNoiseSimsExecutor(BaseStageExecutor):
    """
    GetPlanckNoiseSimsExecutor downloads Planck noise simulation maps.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='get_planck_noise_sims')

        self.out_noise_sim: Asset = self.assets_out['noise_sims']
        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        # For reference:
        in_noise_sim: HealpyMap
        in_det_table_handler: QTableHandler

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.n_sims = cfg.model.sim.noise.n_planck_noise_sims

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        combos = [(det, sim_num) for det in self.instrument.dets for sim_num in range(self.n_sims)]
        n_dets = len(self.instrument.dets)
        with self.name_tracker.set_context('filename', 'dummy_fn'):
            noise_sims_fn = self.out_noise_sim.path
        noise_sims_dir = noise_sims_fn.parent
        with tqdm(total=n_dets*self.n_sims, 
                  desc="Getting Maps", 
                  position=0,
                  dynamic_ncols=True
                  ) as outer_bar:
            for det, sim_num in combos:
                # fn = get_planck_noise_fn(det, sim_num)
                get_planck_noise_data_ext(detector=det, 
                                      assets_directory=noise_sims_dir, 
                                      realization=sim_num, 
                                      progress=True)
                outer_bar.update(1)
        logger.debug("All maps acquired!")
