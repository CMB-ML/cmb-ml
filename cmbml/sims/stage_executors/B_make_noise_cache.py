from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.sims.physics_instrument.registry_noise import get_noise_class

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler


logger = logging.getLogger(__name__)


class NoiseCacheExecutor(BaseStageExecutor):
    """
    NoiseCacheExecutor is responsible for generating and caching noise maps for a given simulation scenario.

    Attributes:
        out_noise_cache (Asset): The output asset for the noise cache.
        in_noise_src (Asset): The input asset for the noise source maps.
        instrument (Instrument): The instrument configuration used for the simulation.

    Methods:
        execute() -> None:
            Executes the noise cache generation process.
        write_wrapper(data: Quantity, field_str: str) -> None:
            Writes the noise map data to the output cache with appropriate column names and units.
        get_field_idx(src_path: str, field_str: str) -> int:
            Determines the field index corresponding to the given field string from the FITS file.
        get_src_path(detector: int) -> str:
            Retrieves the path for the source noise file based on the configuration.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_noise_cache')

        self.noise_maker = get_noise_class(cfg.model.sim.noise.noise_type)

        # For most kinds of noise, we need to cache some values which describe
        # the noise properties. For a few, we do not. `do_cache` indicates this.
        self.do_cache = hasattr(self.noise_maker, "cache_maker") 

        if self.do_cache is False:
            self.out_scale_cache: Asset = None
            self.in_varmap_src: Asset = None
            in_det_table: Asset = None
        else:
            self.out_scale_cache: Asset = self.assets_out['scale_cache']
            self.in_varmap_src: Asset = self.assets_in['noise_src_varmaps']
            # For reference:
            out_noise_cache_handler: HealpyMap
            in_noise_src_handler: HealpyMap

        if self.do_cache is False:
            return

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.CacheMaker_class = self.noise_maker.cache_maker

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        if self.do_cache is False:
            logger.debug(f"No cache being written, because {self.noise_maker.__class__.__name__}.do_cache is False.")
            return
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        cache_maker = self.CacheMaker_class(
                                            cfg=self.cfg, 
                                            name_tracker=self.name_tracker, 
                                            in_varmap_source=self.in_varmap_src,
                                            out_scale_cache=self.out_scale_cache
                                            )
        hdu = self.cfg.model.sim.noise.hdu_n
        for freq, detector in self.instrument.dets.items():
            cache_maker.make_cache_for_freq(freq, detector, hdu)
