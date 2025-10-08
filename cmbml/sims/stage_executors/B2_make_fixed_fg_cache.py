import logging

from omegaconf import DictConfig, OmegaConf

import pysm3
import pysm3.units as u
from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import (
    BaseStageExecutor,
    Asset
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints


logger = logging.getLogger(__name__)


class FixedForegroundExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="make_fixed_fg")

        self.out_fg_map: Asset = self.assets_out['fg_maps']
        out_map_handler: HealpyMap

        in_det_table: Asset = self.assets_in['deltabandpass']
        in_det_table_handler: QTableHandler

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.nside_sky = self.get_nside_sky()
        sky_unit = cfg.model.sim.sky_unit  # Pretty sure it needs to be MJy/sr
        self.sky_unit = u.Unit(sky_unit)
        self.preset_strings = OmegaConf.to_container(cfg.model.sim.preset_strings, resolve=True)
        self.use_fixed_fg = cfg.model.sim.get("use_fixed_fg", None)

    def execute(self) -> None:
        if self.use_fixed_fg is None:
            return
        sky = pysm3.Sky(nside=self.nside_sky, 
                        preset_strings=self.preset_strings,
                        output_unit=self.sky_unit)
        for det in self.instrument.dets.values():
            logger.info(f"Producing map for {det.nom_freq} GHz.")
            skymap = sky.get_emission(det.cen_freq)

            n_fields_sky = skymap.shape[0]
            n_fields_det = len(det.fields)
            if n_fields_sky == n_fields_det:
                pass
            elif n_fields_sky == 3 and n_fields_det == 1:
                # PySM3 components always include T, Q, U; extract the temperature map
                skymap = skymap[0]

            with self.name_tracker.set_context("freq", det.nom_freq):
                self.out_fg_map.write(data=skymap)

    def get_nside_sky(self):
        """
        Returns the nside to use for PySM3's sky object. May be set with one of two 
        configuration options.
        """
        nside_out = self.cfg.scenario.nside
        nside_sky_set = self.cfg.model.sim.get("nside_sky", None)
        nside_sky_factor = self.cfg.model.sim.get("nside_sky_factor", None)

        nside_sky = nside_sky_set if nside_sky_set else nside_out * nside_sky_factor
        return nside_sky
