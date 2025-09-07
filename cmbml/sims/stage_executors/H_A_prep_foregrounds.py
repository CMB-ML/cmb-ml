from typing import Union
import logging

from omegaconf import DictConfig
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u


from cmbml.core import (
    BaseStageExecutor,
    Asset
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.ps_handler import CambPowerSpectrum, NumpyPowerSpectrum # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints

from cmbml.utils.planck_instrument import make_instrument, Instrument, Detector
from cmbml.utils.physics_mean_inpaint import inpaint_with_neighbor_mean


logger = logging.getLogger(__name__)


class PrepForegroundsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_fgs')

        self.out_foregrounds: Asset = self.assets_out['fg_maps']
        out_foregrounds_handler: HealpyMap

        self.in_obs_maps: Asset = self.assets_in['src_obs_maps']
        self.in_cmb_map:  Asset = self.assets_in['src_cmb_map']

        in_pl_det_table:  Asset = self.assets_in['planck_deltabandpass']
        in_sim_det_table: Asset = self.assets_in['sim_deltabandpass']
        in_noise_cache_handler: Union[HealpyMap, NumpyPowerSpectrum]
        in_cmb_ps_handler: CambPowerSpectrum
        in_det_table_handler: QTableHandler

        # Initialize constants from configs
        pl_det_info = in_pl_det_table.read()
        self.pl_instrument: Instrument = make_instrument(cfg=cfg, det_info=pl_det_info)
        sim_det_info = in_sim_det_table.read()
        self.sim_instrument: Instrument = make_instrument(cfg=cfg, det_info=sim_det_info)
        
        self.sky_unit       = u.Unit(cfg.model.sim.sky_unit)
        self.cmb_beam_fwhm  = cfg.model.sim.cmb_beam_fwhm * u.arcmin
        self.sky_nside      = cfg.model.sim.nside_sky

        self.src_files_dict = cfg.model.sim.noise.src_files
        self.assets_dir     = cfg.local_system.assets_dir

        self.inpaint_iter   = cfg.model.sim.inpaint_iters
        self.lmax_ratio     = cfg.model.sim.planck_lmax_ratio
        self.alm_iter_max   = cfg.model.sim.alm_iter_max
        self.alm_iter_tol   = cfg.model.sim.alm_iter_tol
        self.beam_eps       = cfg.model.sim.beam_eps  # avoid instability with wide beam

        # self.cmb_map        = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method")

        self.check_files_exist()
        self.setup()

        for freq in self.pl_instrument.dets.keys():
            fn = self.src_files_dict[freq]
            context = dict(freq=freq, filename=fn)
            with self.name_tracker.set_contexts(context):
                fg_only_map = self.process_order(freq)
                col_names = ["I_FG_ONLY"]
                self.out_foregrounds.write(data=fg_only_map, column_names=col_names)
            logger.info(f"Processed frequency {freq} for foregrounds")

    def check_files_exist(self):
        if not self.in_cmb_map.path.exists():
            raise FileNotFoundError(f"CMB map file not found: {self.in_cmb_map.path}")

        for freq in self.pl_instrument.dets.keys():
            fn = self.src_files_dict[freq]
            context = dict(freq=freq, filename=fn)
            with self.name_tracker.set_contexts(context):
                obs_path = self.in_obs_maps.path
                if not obs_path.exists():
                    raise FileNotFoundError(f"Observation map file not found for frequency {freq}: {obs_path}")

    def setup(self):
        if self.sky_nside in [128, 256]:
            logger.info(f"Downgrading inputs to output at {self.sky_nside}")
            self.process_order = self.process_freq_lowres_sky
        elif self.sky_nside in [1024, 2048]:
            logger.info(f"Downgrading inputs to output at {self.sky_nside}")
            self.process_order = self.process_freq_highres_sky_OLD
        else:
            raise NotImplementedError("This is an untested case.")

    def process_freq_lowres_sky(self, freq: int):
        """
        Sequentially processes maps in alm space with maximal appropriate SHT
        To be used when resolution of CMB >= Obs > Sky.
        E.g., when making simulations at 128:
                         CMB prediction at 2048, 
                         Observations at either 2048 or 1024, 
                         Sky at 256 
                         (simulations at 128, downgraded in subsequent executor)
        May apply to other cases.
        """

        # "Convolve" means deconvolve from some source beam and reconvolve
        # Get maps and associated parameters & beams
        # Set up sky pixwin
        # Set up sky beam; note that this will both convolve and CUT high-ell signal
        # CMB: convolve with sky beam and obs pixwin; downgrade to obs reso
        # Obs: convolve with sky beam
        # Get difference map
        # Remove difference map pixwin; downgrade to sky resolution

        lmax_ratio = self.lmax_ratio
        inpaint_iter = self.inpaint_iter
        beam_eps = self.beam_eps
        sky_unit = self.sky_unit

        src_det:Detector = self.pl_instrument.dets[freq]
        sky_det:Detector = self.sim_instrument.dets[freq]

        # Get cmb map and parameters
        cmb_map = self.in_cmb_map.read()[0]
        cmb_map = cmb_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        cmb_nside = hp.get_nside(cmb_map)
        cmb_lmax = int(lmax_ratio * cmb_nside)
        # Note: use obs_lmax in next lines; this makes array sizes match
        cmb_beam_fwhm = self.cmb_beam_fwhm.to(u.rad).value
        cmb_beam = hp.gauss_beam(cmb_beam_fwhm, lmax=cmb_lmax)
        cmb_pixwin = hp.pixwin(cmb_nside, lmax=cmb_lmax, pol=False)

        # Get observation map and parameters
        obs_map = self.in_obs_maps.read()[0]
        obs_nside = hp.get_nside(obs_map)
        obs_lmax = int(lmax_ratio * obs_nside)

        obs_map = inpaint_with_neighbor_mean(obs_map, inpaint_iter)  # Fix (single) UNSEEN pixel in 100GHz map with average of neighbors
        obs_map = obs_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        obs_beam_fwhm = src_det.fwhm.to(u.rad).value
        obs_beam = hp.gauss_beam(obs_beam_fwhm, obs_lmax)
        obs_pixwin = np.zeros_like(cmb_pixwin)
        obs_pixwin[:obs_lmax+1] = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

        # Sky prep
        sky_nside = self.sky_nside
        sky_lmax = int(lmax_ratio * sky_nside)
        sky_pixwin = hp.pixwin(nside=sky_nside, lmax=sky_lmax)

        sky_beam_fwhm = sky_det.fwhm
        sky_beam_fwhm = sky_beam_fwhm.to(u.rad).value
        sky_beam = hp.gauss_beam(sky_beam_fwhm, cmb_lmax)

        # Process CMB map: operate at CMB lmax (alm space) (highest resolution)
        #                  beam             : cmb -> sky
        #                  pixwin           : cmb -> obs
        #                  output resolution: cmb -> obs
        zero_ells = np.ones_like(cmb_beam)
        zero_ells[obs_lmax+1:] = 0
        cmb_fl = (zero_ells * sky_beam * obs_pixwin) / (cmb_beam * cmb_pixwin)
        cmb_map_bmd = self.apply_fl_in_sht(cmb_map, cmb_fl, obs_nside)
        logger.info(f"Made rebeamed CMB map for {freq} GHz.")

        # Process obs map: operate at Obs lmax (alm space) (possibly lower resolution, e.g. LFI)
        #                  beam             : obs -> sky
        #                  pixwin           : obs -> obs (no change)
        #                  output resolution: obs -> obs (no change)
        obs_fl = obs_beam / (obs_beam**2 + beam_eps)  # safe way to handle wide (e.g., 30GHz) obs beams
        obs_fl = obs_fl * sky_beam[:obs_lmax+1]
        obs_map_bmd = self.apply_fl_in_sht(obs_map, obs_fl, obs_nside)
        logger.info(f"Made rebeamed Obs map for {freq} GHz.")
        
        diff_map_hr = obs_map_bmd - cmb_map_bmd  # hr = high resolution (obs not sky)

        # Process diff map: operate at Obs lmax (alm space) (lowest resolution)
        #                   beam             : sky -> sky (no change)
        #                   pixwin           : obs -> sky
        #                   output resolution: obs -> sky
        diff_fl = np.zeros_like(obs_beam)
        diff_fl[:sky_lmax+1] = sky_pixwin / obs_pixwin[:sky_lmax+1]
        diff_map = self.apply_fl_in_sht(diff_map_hr, diff_fl, sky_nside)

        diff_map = u.Quantity(diff_map, self.sky_unit)
        return diff_map

    def apply_fl_in_sht(self, in_map, fl, out_nside):
        in_nside = hp.get_nside(in_map)
        in_lmax = fl.shape[0]
        if in_lmax != in_nside * self.lmax_ratio + 1:
            raise ValueError(f"in_lmax: {in_lmax}, expected {in_nside * self.lmax_ratio} in_nside: {in_nside}, ratio: {self.lmax_ratio}")
        out_lmax = int(out_nside * self.lmax_ratio + 1)
        alms = self.map2alm(in_map, in_lmax)
        alms_bmd = hp.almxfl(alms, fl)
        alms_bmd = hp.resize_alm(alms_bmd, lmax=in_lmax, mmax=in_lmax, lmax_out=out_lmax, mmax_out=out_lmax)
        map_bmd = hp.alm2map(alms_bmd, out_nside, out_lmax)
        return map_bmd

    def map2alm(self, in_map, lmax):
        """
        Here for clarity. Based on PySM3 (copied? possibly modified.)
        """
        if self.alm_iter_max == 0:
            logger.info("Using map2alm without weights and no iterations.")
            alms = hp.map2alm(in_map, lmax=lmax, iter=self.alm_iter_max, pol=False)
        else:
            alms, error, n_iter = hp.map2alm_lsq(
                maps=in_map,
                lmax=lmax,
                mmax=lmax,
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
                    + "residual relative error %.2g",  # I do not know why the observation maps "converge" after some iterations with error greater than alm_tol_iter
                    n_iter,
                    error,
                )
        return alms

    def process_freq_2048_1024_sky(self, freq: int):
        """
        Sequentially processes maps in alm space with maximal appropriate SHT
        Assume CMB at 2048; 
               sky at 2048; 
               NO ~~~~sky at either 1024 or 2048; ~~~
               observations at either 1024 or 2048
        To be used when resolution of CMB >= Sky >= Obs.
        E.g., when making simulations at 512:
                         CMB prediction at 2048, 
                         Observations at either 2048 or 1024, 
                         Sky at 2048 
                         (simulations at 512, downgraded in other Executor)
        May apply to other cases.
        """

        lmax_ratio = self.lmax_ratio
        alm_max_iter = self.alm_iter_max
        inpaint_iter = self.inpaint_iter
        beam_eps = self.beam_eps
        sky_unit = self.sky_unit

        src_det:Detector      = self.pl_instrument.dets[freq]
        sky_det:Detector      = self.sim_instrument.dets[freq]

        # Get cmb map and parameters
        cmb_map               = self.in_cmb_map.read()[0]
        cmb_map               = cmb_map.to(sky_unit,
                                           equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        cmb_nside             = hp.get_nside(cmb_map)
        cmb_lmax              = int(lmax_ratio * cmb_nside)
        cmb_beam_fwhm         = self.cmb_beam_fwhm
        cmb_beam              = hp.gauss_beam(cmb_beam_fwhm.to(u.rad).value, lmax=cmb_lmax)
        cmb_pxwn              = hp.pixwin(cmb_nside, lmax=cmb_lmax, pol=False)

        # Sky prep (Don't need to worry about minimum beam size)
        sky_lmax              = int(self.lmax_ratio * self.sky_nside)
        sky_beam_fwhm         = sky_det.fwhm.to(u.rad).value
        sky_beam              = hp.gauss_beam(sky_beam_fwhm, lmax=cmb_lmax)
        sky_pxwn              = np.zeros_like(cmb_pxwn)
        sky_pxwn[:sky_lmax+1] = hp.pixwin(nside=self.sky_nside, lmax=sky_lmax, pol=False)

        # Get observation map and parameters (Use CMB lmax)
        obs_map       = self.in_obs_maps.read()[0]
        obs_nside     = hp.get_nside(obs_map)
        obs_lmax      = int(lmax_ratio * obs_nside)

        obs_map       = inpaint_with_neighbor_mean(obs_map, inpaint_iter)
        obs_map       = obs_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        obs_beam_fwhm = src_det.fwhm.to(u.rad).value
        obs_beam      = hp.gauss_beam(obs_beam_fwhm, obs_lmax)
        obs_pixwin    = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

        # Process CMB map: operate at CMB lmax (alm space) (highest resolution)
        #                  Note: need to exclude high frequency information to match obs (?)
        #                  beam             : cmb -> sky
        #                  pixwin           : cmb -> cmb (no change)
        #                  output resolution: cmb -> cmb (no change)
        cmb_alms      = pysm3.map2alm(input_map=cmb_map,
                                      nside=None,
                                      lmax=cmb_lmax,
                                      map2alm_lsq_maxiter=alm_max_iter)
        logger.info(f"Made rebeamed CMB map for {freq} GHz.")

        # For excluding (muting) high-ell information between obs_lmax and cmb_lmax
        mute_fl              = np.zeros(cmb_lmax+1)
        mute_fl[:obs_lmax+1] = 1

        cmb_fl               = mute_fl * sky_beam / cmb_beam
        cmb_alms_bmd         = hp.almxfl(cmb_alms, cmb_fl)
        cmb_map_bmd          = hp.alm2map(cmb_alms_bmd, nside=cmb_nside)

        # Process obs map: operate at Obs lmax (alm space) (possibly lower resolution, e.g. LFI)
        #                  beam             : obs -> sky
        #                  pixwin           : obs -> cmb
        #                  output resolution: obs -> cmb
        obs_alms       = pysm3.map2alm(input_map=obs_map,
                                       nside=None,
                                       lmax=obs_lmax,
                                       map2alm_lsq_maxiter=alm_max_iter)
        sky_beam_l_obs = sky_beam[:obs_lmax+1]    # sky beam with length of obs_lmax
        cmb_pxwn_l_obs = cmb_pxwn[:obs_lmax+1]  # cmb pixel window with length of obs_lmax
        # Build obs fl in two steps; 
        #   Step 1 produces a safe beam that doesn't explode due to small values
        obs_fl         = obs_beam / (obs_beam**2 + beam_eps)
        obs_fl         = sky_beam_l_obs * cmb_pxwn_l_obs * obs_fl / obs_pixwin
        obs_alms_bmd   = hp.almxfl(obs_alms, obs_fl)
        obs_map_bmd    = hp.alm2map(obs_alms_bmd, nside=cmb_nside)
        logger.info(f"Made rebeamed Obs map for {freq} GHz.")

        diff_map       = obs_map_bmd - cmb_map_bmd

        # Process diff map: operate at CMB lmax (alm space) (lowest resolution)
        #                   beam             : sky -> sky (no change)
        #                   pixwin           : cmb -> sky
        #                   output resolution: cmb -> sky
        diff_alms      = pysm3.map2alm(input_map=diff_map,
                                       nside=None,
                                       lmax=cmb_lmax,
                                       map2alm_lsq_maxiter=alm_max_iter)
        diff_fl        = sky_pxwn / cmb_pxwn
        diff_alms_bmd  = hp.almxfl(diff_alms, diff_fl)
        diff_map_bmd   = hp.alm2map(diff_alms_bmd, nside=self.sky_nside)

        diff_map_bmd   = u.Quantity(diff_map, self.sky_unit)
        return diff_map_bmd

    def process_freq_highres_sky_OLD(self, freq: int):
        lmax_ratio = self.lmax_ratio
        alm_max_iter = self.alm_iter_max
        inpaint_iter = self.inpaint_iter
        beam_eps = self.beam_eps
        sky_unit = self.sky_unit
        src_det:Detector = self.pl_instrument.dets[freq]
        sky_det:Detector = self.sim_instrument.dets[freq]

        # Get observation map and parameters
        obs_map = self.in_obs_maps.read()[0]
        obs_nside = hp.get_nside(obs_map)
        obs_lmax = int(lmax_ratio * obs_nside)

        obs_map = inpaint_with_neighbor_mean(obs_map, inpaint_iter)
        obs_map = obs_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        obs_beam_fwhm = src_det.fwhm.to(u.rad).value
        obs_beam = hp.gauss_beam(obs_beam_fwhm, obs_lmax)
        obs_pixwin = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

        # Get cmb map and parameters (USE lmax for CMB!)
        cmb_map = self.in_cmb_map.read()[0]
        cmb_map = cmb_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        cmb_nside = hp.get_nside(cmb_map)
        cmb_lmax = int(lmax_ratio * cmb_nside)
        cmb_beam_fwhm = self.cmb_beam_fwhm
        cmb_beam = hp.gauss_beam(cmb_beam_fwhm.to(u.rad).value, lmax=cmb_lmax)
        cmb_pixwin = hp.pixwin(cmb_nside, lmax=cmb_lmax, pol=False)

        # Sky prep (Don't need to worry about minimum beam size)
        sky_lmax       = int(self.lmax_ratio * self.sky_nside)
        sky_beam_fwhm  = sky_det.fwhm.to(u.rad).value
        # While running a cross-check with physics, I needed to confirm that my changes to this method (above)
        #   didn't alter output maps. Assumptions made caused sky stuff to have different vector shape; this fixes it
        #   Note that it's different from original, but original was broken. 
        #   Eventually both this old version and the true one (commented out below) will be deleted.
        sky_beam                = hp.gauss_beam(sky_beam_fwhm, lmax=cmb_lmax)
        sky_beam[sky_lmax+1:]   = 0
        sky_pixwin              = np.zeros_like(cmb_pixwin)
        sky_pixwin[:sky_lmax+1] = hp.pixwin(nside=self.sky_nside, lmax=sky_lmax, pol=False)

        # Get observations at sky beam and cmb Nside (assumes reso_obs <= reso_cmb)
        obs_alms = pysm3.map2alm(input_map=obs_map,
                                 nside=None,
                                 lmax=obs_lmax,
                                 map2alm_lsq_maxiter=alm_max_iter)
        sky_beam_l_obs = sky_beam[:obs_lmax+1]
        cmb_pxwn_l_obs = cmb_pixwin[:obs_lmax+1]
        # Build obs fl in two steps; 
        #   Step 1 produces a safe beam that doesn't explode due to small values
        obs_fl = obs_beam / (obs_beam**2 + beam_eps)
        obs_fl = sky_beam_l_obs * cmb_pxwn_l_obs * obs_fl / obs_pixwin
        obs_alms_bmd = hp.almxfl(obs_alms, obs_fl)
        obs_map_bmd = hp.alm2map(obs_alms_bmd, nside=cmb_nside)

        # Get cmb at sky beam (and cmb nside)
        cmb_alms = pysm3.map2alm(input_map=cmb_map,
                                 nside=None,
                                 lmax=cmb_lmax,
                                 map2alm_lsq_maxiter=alm_max_iter)
        sky_beam_l_cmb = sky_beam[:cmb_lmax+1]
        cmb_fl = sky_beam_l_cmb / cmb_beam
        cmb_alms_bmd = hp.almxfl(cmb_alms, cmb_fl)
        cmb_map_bmd = hp.alm2map(cmb_alms_bmd, nside=cmb_nside)

        diff_map = obs_map_bmd - cmb_map_bmd

        # Convert diff to sky nside
        diff_alms = pysm3.map2alm(input_map=diff_map,
                                  nside=None,
                                  lmax=cmb_lmax,
                                  map2alm_lsq_maxiter=alm_max_iter)
        sky_pxwn_l_cmb = sky_pixwin[:cmb_lmax+1]
        diff_fl = sky_pxwn_l_cmb / cmb_pixwin
        diff_alms_bmd = hp.almxfl(diff_alms, diff_fl)
        diff_map_bmd = hp.alm2map(diff_alms_bmd, nside=self.sky_nside)

        diff_map_bmd = u.Quantity(diff_map, self.sky_unit)
        return diff_map_bmd

    # True original
    # def process_freq_highres_sky_OLD(self, freq: int):
    #     lmax_ratio = self.lmax_ratio
    #     alm_max_iter = self.alm_iter_max
    #     inpaint_iter = self.inpaint_iter
    #     beam_eps = self.beam_eps
    #     sky_unit = self.sky_unit
    #     src_det:Detector = self.pl_instrument.dets[freq]
    #     sky_det:Detector = self.sim_instrument.dets[freq]

    #     # Get observation map and parameters
    #     obs_map = self.in_obs_maps.read()[0]
    #     obs_nside = hp.get_nside(obs_map)
    #     obs_lmax = int(lmax_ratio * obs_nside)

    #     obs_map = inpaint_with_neighbor_mean(obs_map, inpaint_iter)
    #     obs_map = obs_map.to(sky_unit,
    #                          equivalencies=u.cmb_equivalencies(src_det.cen_freq))
    #     obs_beam_fwhm = src_det.fwhm.to(u.rad).value
    #     obs_beam = hp.gauss_beam(obs_beam_fwhm, obs_lmax)
    #     obs_pixwin = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

    #     # Get cmb map and parameters (USE lmax for CMB!)
    #     cmb_map = self.in_cmb_map.read()[0]
    #     cmb_map = cmb_map.to(sky_unit,
    #                          equivalencies=u.cmb_equivalencies(src_det.cen_freq))
    #     cmb_nside = hp.get_nside(cmb_map)
    #     cmb_lmax = int(lmax_ratio * cmb_nside)
    #     cmb_beam_fwhm = self.cmb_beam_fwhm
    #     cmb_beam = hp.gauss_beam(cmb_beam_fwhm.to(u.rad).value, lmax=cmb_lmax)
    #     cmb_pixwin = hp.pixwin(cmb_nside, lmax=cmb_lmax, pol=False)

    #     # Sky prep (Don't need to worry about minimum beam size)
    #     sky_lmax       = int(self.lmax_ratio * self.sky_nside)
    #     sky_beam_fwhm  = sky_det.fwhm.to(u.rad).value
    #     sky_beam       = hp.gauss_beam(sky_beam_fwhm, lmax=sky_lmax)
    #     sky_pixwin     = hp.pixwin(nside=self.sky_nside, lmax=sky_lmax, pol=False)

    #     # Get observations at sky beam and cmb Nside (assumes reso_obs <= reso_cmb)
    #     obs_alms = pysm3.map2alm(input_map=obs_map,
    #                              nside=None,
    #                              lmax=obs_lmax,
    #                              map2alm_lsq_maxiter=alm_max_iter)
    #     sky_beam_l_obs = sky_beam[:obs_lmax+1]
    #     cmb_pxwn_l_obs = cmb_pixwin[:obs_lmax+1]
    #     # Build obs fl in two steps; 
    #     #   Step 1 produces a safe beam that doesn't explode due to small values
    #     obs_fl = obs_beam / (obs_beam**2 + beam_eps)
    #     obs_fl = sky_beam_l_obs * cmb_pxwn_l_obs * obs_fl / obs_pixwin
    #     obs_alms_bmd = hp.almxfl(obs_alms, obs_fl)
    #     obs_map_bmd = hp.alm2map(obs_alms_bmd, nside=cmb_nside)

    #     # Get cmb at sky beam (and cmb nside)
    #     cmb_alms = pysm3.map2alm(input_map=cmb_map,
    #                              nside=None,
    #                              lmax=cmb_lmax,
    #                              map2alm_lsq_maxiter=alm_max_iter)
    #     sky_beam_l_cmb = sky_beam[:cmb_lmax+1]
    #     cmb_fl = sky_beam_l_cmb / cmb_beam
    #     cmb_alms_bmd = hp.almxfl(cmb_alms, cmb_fl)
    #     cmb_map_bmd = hp.alm2map(cmb_alms_bmd, nside=cmb_nside)

    #     diff_map = obs_map_bmd - cmb_map_bmd

    #     # Convert diff to sky nside
    #     diff_alms = pysm3.map2alm(input_map=diff_map,
    #                               nside=None,
    #                               lmax=cmb_lmax,
    #                               map2alm_lsq_maxiter=alm_max_iter)
    #     sky_pxwn_l_cmb = sky_pixwin[:cmb_lmax+1]
    #     diff_fl = sky_pxwn_l_cmb / cmb_pixwin
    #     diff_alms_bmd = hp.almxfl(diff_alms, diff_fl)
    #     diff_map_bmd = hp.alm2map(diff_alms_bmd, nside=self.sky_nside)

    #     diff_map_bmd = u.Quantity(diff_map, self.sky_unit)
    #     return diff_map_bmd
