from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path
import healpy as hp
import pysm3
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset
# from cmbml.utils.planck_instrument import make_instrument, Instrument
# from cmbml.sims.physics_instrument import get_noise_class

# from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
# from cmbml.core.asset_handlers.qtable_handler import QTableHandler
# import cmbml.utils.fits_inspection as fits_inspect
# from cmbml.utils.physics_units import convert_field_str_to_Unit
# from cmbml.utils.fits_inspection import get_num_all_fields_in_hdr


logger = logging.getLogger(__name__)


class CMBPSConvertExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str='convert_ps')

        self.out_cmb_ps: Asset = self.assets_out['cmb_ps']
        self.in_cmb_ps: Asset = self.assets_in['cmb_ps']


    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        in_cmb_ps = self.in_cmb_ps.read()
        out_cmb_ps = in_cmb_ps.split('\n')
        
        # Insert placeholder monopole and dipole
        pl_mono = "     0   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00"
        pl_di   = "     1   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00"
        out_cmb_ps.insert(1, pl_mono)
        out_cmb_ps.insert(2, pl_di)
        out_cmb_ps = '\n'.join(out_cmb_ps)

        context = dict(
            split="Test",
            sim_num=0
        )
        with self.name_tracker.set_contexts(context):
            self.out_cmb_ps.write(data=out_cmb_ps)

