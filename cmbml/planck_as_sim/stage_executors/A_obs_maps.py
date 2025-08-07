from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument, Detector

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.utils.physics_mean_inpaint import inpaint_with_neighbor_mean


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
        self.out_instrument: Instrument = make_instrument(
            cfg=cfg, det_info=det_info
            )
        self.planck_instrument: Instrument = make_instrument(
            cfg=cfg, det_info=planck_det_info, use_min_fwhm=False
            )

        # Most of the code here is pulled from the NoiseCacheExecutor, which
        #   uses the same maps. TODO: Generalize this.
        self.obs_files = cfg.model.sim.noise.src_files
        self.obs_root = cfg.local_system.assets_dir
        self.hdu = self.cfg.model.sim.noise.hdu_n
        # self.field_idcs = {3: {'I': 0}, 10: {'I': 0, 'Q': 1, 'U': 2}}
        self.out_nside = cfg.scenario.nside
        # Use fixed value; we are downgrading the Planck maps.
        self.sky_unit = u.Unit(cfg.model.sim.sky_unit)
        self.out_unit = cfg.scenario.units

        self.inpaint_iter   = cfg.model.sim.inpaint_iters
        self.lmax_ratio     = cfg.model.sim.planck_lmax_ratio
        self.alm_iter_max   = cfg.model.sim.alm_iter_max
        self.alm_iter_tol   = cfg.model.sim.alm_iter_tol
        self.beam_eps       = cfg.model.sim.beam_eps  # avoid instability with wide beam

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        for freq, detector in self.out_instrument.dets.items():
            logger.info(f"Processing frequency {freq} GHz for Stokes {detector.fields}")
            smoothed_map = self.process_freq(freq)

            # Write the smoothed map to the output asset
            context = dict(
                split='Test',
                sim_num=0,
                freq=freq
            )
            with self.name_tracker.set_contexts(context):
                self.out_obs_maps.write(data=smoothed_map)

    def get_obs_path(self, freq) -> Path:
        fn = self.obs_files[freq]
        obs_root = self.obs_root
        context_dict = dict(src_root=obs_root, filename=fn)
        with self.name_tracker.set_contexts(context_dict):
            src_path = self.in_obs_maps.path
        return src_path
    
    def process_freq(self, freq):
        plk_det: Detector = self.planck_instrument.dets[freq]
        out_det: Detector = self.out_instrument.dets[freq]
        obs_path = self.get_obs_path(freq)
        # field_idcs = [self.get_field_idx(src_path, field_str) for field_str in detector.fields]
        # obs_unit = fits_inspect.get_field_unit_str(src_path, field_idcs[0], hdu=self.hdu)
        # obs_unit = convert_field_str_to_Unit(obs_unit)
        # obs_map = hp.read_map(src_path, hdu=self.hdu, field=field_idcs)

        h = HealpyMap()
        if len(plk_det.fields) > 1:
            raise NotImplementedError("Can't handle polarization.")
        obs_map = h.read(path=obs_path)[0]
        obs_nside = hp.get_nside(obs_map)
        obs_lmax = int(self.lmax_ratio * obs_nside)

        obs_map = inpaint_with_neighbor_mean(obs_map, self.inpaint_iter)
        obs_map = obs_map.to(self.sky_unit, 
                             equivalencies=u.cmb_equivalencies(plk_det.cen_freq))
        obs_beam_fwhm = plk_det.fwhm.to(u.rad).value
        obs_beam = hp.gauss_beam(obs_beam_fwhm, obs_lmax)
        obs_pxwn = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

        out_lmax = int(self.lmax_ratio * self.out_nside)
        out_beam_fwhm = out_det.fwhm.to(u.rad).value

        # Bandwidth limit
        out_beam = np.zeros_like(obs_beam)
        out_beam[:out_lmax+1] = hp.gauss_beam(out_beam_fwhm, out_lmax)
        out_pxwn = np.zeros_like(obs_pxwn)
        out_pxwn[:out_lmax+1] = hp.pixwin(nside=self.out_nside, lmax=out_lmax, pol=False)

        # From PySM3 map2alm (copied due to issues and troubleshooting)
        # TODO: Return to using pysm3.map2alm()
        if self.alm_iter_max == 0:
            logger.info("Using map2alm without weights and no iterations.")
            obs_alms = hp.map2alm(obs_map, lmax=obs_lmax, iter=self.alm_iter_max, pol=False)
        else:
            obs_alms, error, n_iter = hp.map2alm_lsq(
                maps=obs_map,
                lmax=obs_lmax,
                mmax=obs_lmax,
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

        safe_debeam_fl = obs_beam / (obs_beam**2 + self.beam_eps)
        fl = out_beam * out_pxwn * safe_debeam_fl / obs_pxwn
        out_alms = hp.almxfl(obs_alms, fl)
        out_map = hp.alm2map(out_alms, self.out_nside)
        out_map = u.Quantity(out_map, self.sky_unit)

        out_map = out_map.to(self.out_unit,
                             equivalencies=u.cmb_equivalencies(plk_det.cen_freq))

        return out_map
