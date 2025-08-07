from typing import Dict
import pysm3
import logging

from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset, HealpyMap
from cmbml.utils.planck_instrument import make_instrument, Instrument
# from cmbml.sims.physics_instrument import get_noise_class

# from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
# from cmbml.core.asset_handlers.qtable_handler import QTableHandler
import cmbml.utils.fits_inspection as fits_inspect
from cmbml.utils.physics_units import convert_field_str_to_Unit
from cmbml.utils.fits_inspection import get_num_all_fields_in_hdr
from cmbml.utils.physics_mean_inpaint import inpaint_with_neighbor_mean


logger = logging.getLogger(__name__)


class CMBMapConvertExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str='convert_cmb')

        self.out_cmb_map: Asset = self.assets_out['cmb_map']

        in_det_table: Asset = self.assets_in['deltabandpass']
        in_planck_det_table: Asset = self.assets_in['planck_deltabandpass']
        self.in_cmb_map: Asset = self.assets_in['cmb_map']

        # with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
        #     det_info = in_det_table.read()
        # self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)
        
        # Default beam size for CMB maps
        # https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_maps
        self.in_beam_fwhm = cfg.model.sim.in_cmb_beam_fwhm * u.arcmin
        self.out_beam_fwhm = cfg.model.sim.out_cmb_beam_fwhm * u.arcmin

        self.hdu = self.cfg.model.sim.noise.hdu_n
        # Hard-coding for temperature only maps.
        self.field_idx = 0
        self.nside_out = cfg.scenario.nside
        # Use fixed value; we are downgrading the Planck maps.
        self.out_unit       = cfg.scenario.units
        self.lmax_ratio     = cfg.model.sim.planck_lmax_ratio

        self.inpaint_iter   = cfg.model.sim.inpaint_iters
        self.lmax_ratio     = cfg.model.sim.planck_lmax_ratio
        self.alm_iter_max   = cfg.model.sim.alm_iter_max
        self.alm_iter_tol   = cfg.model.sim.alm_iter_tol

        self.cmb_unit       = u.Unit(cfg.model.sim.cmb_unit)

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        smoothed_map = self.process_map()

        # Write the smoothed map to the output asset
        context = dict(split='Test', sim_num=0)
        with self.name_tracker.set_contexts(context):
            self.out_cmb_map.write(data=smoothed_map)
    
    def process_map(self):
        field_idcs = [self.get_field_idx(field_str) for field_str in self.map_fields]
        if len(field_idcs) > 1:
            raise NotImplementedError("Can't handle polarization.")

        src_path = self.in_cmb_map.path
        h = HealpyMap()
        # source_map = hp.read_map(src_path, hdu=self.hdu, field=field_idcs)
        src_map = h.read(src_path)
        src_unit = src_map.unit
        src_nside = hp.get_nside(src_map)
        src_lmax = int(self.lmax_ratio * src_nside)

        out_nside = self.nside_out
        out_lmax = int(self.lmax_ratio * out_nside)

        src_map = inpaint_with_neighbor_mean(src_map, self.inpaint_iter)
        src_map = src_map.to(self.cmb_unit)  # Assume it's K_CMB or uK_CMB

        src_beam_size_rad = self.in_beam_fwhm.to(u.rad).value
        src_beam = hp.gauss_beam(fwhm=src_beam_size_rad, lmax=src_lmax)
        src_pxwn = hp.pixwin(nside=src_nside, lmax=src_lmax, pol=False)

        out_beam_size_rad = self.out_beam_fwhm.to(u.rad).value
        out_beam = np.zeros_like(src_beam)
        out_beam[:out_lmax+1] = hp.gauss_beam(fwhm=out_beam_size_rad, lmax=out_lmax)
        out_pxwn = np.zeros_like(src_pxwn)
        out_pxwn[:out_lmax+1] = hp.pixwin(nside=out_nside, lmax=out_lmax, pol=False)

        beam_ratio = out_beam * out_pxwn / (src_beam * src_pxwn)

        # From PySM3 map2alm (copied due to issues and troubleshooting)
        # TODO: Return to using pysm3.map2alm()
        if self.alm_iter_max == 0:
            logger.info("Using map2alm without weights and no iterations.")
            in_alms = hp.map2alm(src_map, lmax=src_lmax, iter=self.alm_iter_max, pol=False)
        else:
            in_alms, error, n_iter = hp.map2alm_lsq(
                maps=src_map,
                lmax=src_lmax,
                mmax=src_lmax,
                tol=self.alm_iter_tol,
                maxiter=self.alm_iter_max
            )
            if n_iter == self.alm_iter_max:
                logger.warning(
                    "hp.map2alm_lsq did not converge in %d iterations,"
                    + " residual relative error is %.2g",
                    n_iter,
                    error,
                )
            else:
                logger.info(
                    "Used map2alm_lsq, converged in %d iterations,"
                    + "residual relative error %.2g",
                    n_iter,
                    error,
                )

        out_alms = hp.almxfl(in_alms, beam_ratio)
        out_map = hp.alm2map(out_alms, nside=self.nside_out)

        out_map = u.Quantity(out_map, src_unit)
        out_map = out_map.to(self.out_unit)

        return out_map

    def get_field_idx(self, field_str) -> int:
        field_idcs_dict = {
            'I': 0,  # Temperature map
            'Q': 1,  # Stokes Q parameter
            'U': 2   # Stokes U parameter
        }
        return field_idcs_dict[field_str]
