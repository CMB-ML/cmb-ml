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
from cmbml.utils.planck_instrument import make_instrument, Instrument
# from cmbml.sims.physics_instrument import get_noise_class

# from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
# from cmbml.core.asset_handlers.qtable_handler import QTableHandler
import cmbml.utils.fits_inspection as fits_inspect
from cmbml.utils.physics_units import convert_field_str_to_Unit
from cmbml.utils.fits_inspection import get_num_all_fields_in_hdr
from cmbml.utils.physics_downgrade_by_alm import pysm3_downgrade_T_by_alm


logger = logging.getLogger(__name__)


class ObsMapsConvertExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str='convert_obs')

        self.out_obs_maps: Asset = self.assets_out['obs_maps']

        in_det_table: Asset = self.assets_in['deltabandpass']
        in_planck_det_table: Asset = self.assets_in['planck_deltabandpass']
        self.in_obs_maps: Asset = self.assets_in['obs_maps']

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
            planck_det_info = in_planck_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)
        self.planck_instrument: Instrument = make_instrument(
            cfg=cfg, det_info=planck_det_info
        )

        # Most of the code here is pulled from the NoiseCacheExecutor, which
        #   uses the same maps. TODO: Generalize this.
        self.src_files = cfg.model.sim.noise.src_files
        self.src_root = cfg.local_system.assets_dir
        self.hdu = self.cfg.model.sim.noise.hdu_n
        self.field_idcs = {3: {'I': 0}, 10: {'I': 0, 'Q': 1, 'U': 2}}
        self.nside_out = cfg.scenario.nside
        # Use fixed value; we are downgrading the Planck maps.
        self.out_unit = cfg.scenario.units
        self.lmax_ratio = 1.5

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        for freq, detector in self.instrument.dets.items():
            logger.info(f"Processing frequency {freq} GHz for Stokes {detector.fields}")
            smoothed_map = self.process_freq(freq, detector)

            # Write the smoothed map to the output asset
            context = dict(
                split='Test',
                sim_num=0,
                freq=freq
            )
            with self.name_tracker.set_contexts(context):
                self.out_obs_maps.write(data=smoothed_map)

    def get_src_path(self, freq) -> Path:
        fn = self.src_files[freq]
        src_root = self.src_root
        context_dict = dict(src_root=src_root, filename=fn)
        with self.name_tracker.set_contexts(context_dict):
            src_path = self.in_obs_maps.path
        return src_path
    
    def process_freq(self, freq, detector):
        src_path = self.get_src_path(freq)
        field_idcs = [self.get_field_idx(src_path, field_str) for field_str in detector.fields]

        src_unit = fits_inspect.get_field_unit_str(src_path, field_idcs[0], hdu=self.hdu)
        src_unit = convert_field_str_to_Unit(src_unit)

        source_map = hp.read_map(src_path, hdu=self.hdu, field=field_idcs)

        src_beam_fwhm = self.planck_instrument.dets[freq].fwhm
        tgt_beam_fwhm = detector.fwhm

        smoothed_map = pysm3_downgrade_T_by_alm(source_map, self.nside_out, src_beam_fwhm, tgt_beam_fwhm)

        smoothed_map *= src_unit
        smoothed_map = smoothed_map.to(self.out_unit, 
                                       equivalencies=u.cmb_equivalencies(detector.cen_freq))

        return smoothed_map

    def get_field_idx(self, src_path, field_str) -> int:
        """
        Looks at fits file to determine field_idx corresponding to field_str

        Parameters:
        src_path (str): The path to the fits file.
        field_str (str): The field string to look up.

        Returns:
        int: The field index corresponding to the field string.
        """
        field_idcs_dict = dict(self.field_idcs)
        # Get number of fields in map
        n_map_fields = get_num_all_fields_in_hdr(fits_fn=src_path, hdu=self.hdu)
        # Lookup field index based on config file
        field_idx = field_idcs_dict[n_map_fields][field_str]
        return field_idx
