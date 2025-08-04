from typing import Union
import logging

from omegaconf import DictConfig
import healpy as hp
import pysm3
import pysm3.units as u


from cmbml.core import (
    BaseStageExecutor,
    Asset
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.psmaker_handler import CambPowerSpectrum, NumpyPowerSpectrum # Import for typing hint
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
        
        self.sky_unit           = u.Unit(cfg.model.sim.sky_unit)
        self.cmb_beam_fwhm  = cfg.model.sim.cmb_beam_fwhm * u.arcmin
        self.sky_nside      = cfg.model.sim.nside_sky

        self.src_files_dict = cfg.model.sim.noise.src_files
        self.assets_dir     = cfg.local_system.assets_dir

        self.lmax_ratio     = cfg.model.sim.planck_lmax_ratio
        self.inpaint_iter   = cfg.model.sim.inpaint_iters
        self.alm_max_iter   = cfg.model.sim.alm_max_iter
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
        if self.sky_nside == 256:
            logger.info(f"Downgrading inputs to output at {self.sky_nside}")
            self.process_order = self.process_freq_lowres_sky
        elif self.sky_nside == 2048:
            logger.info(f"Downgrading inputs to output at {self.sky_nside}")
            self.process_order = self.process_freq_highres_sky
        else:
            raise NotImplementedError("This is an untested case.")

    def process_freq_lowres_sky(self, freq: int):

        # "Convolve" means deconvolve from some source beam and reconvolve
        # Get maps and associated parameters & beams
        # Set up sky pixwin
        # Set up sky beam; note that this will both convolve and CUT high-ell signal
        # CMB: convolve with sky beam and obs pixwin; downgrade to obs reso
        # Obs: convolve with sky beam
        # Get difference map
        # Remove difference map pixwin; downgrade to sky resolution

        lmax_ratio = self.lmax_ratio
        alm_max_iter = self.alm_max_iter
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

        # Get cmb map and parameters
        cmb_map = self.in_cmb_map.read()[0]
        cmb_map = cmb_map.to(sky_unit,
                             equivalencies=u.cmb_equivalencies(src_det.cen_freq))
        cmb_nside = hp.get_nside(cmb_map)
        cmb_lmax = int(lmax_ratio * cmb_nside)
        # Note: use obs_lmax in next lines; this makes array sizes match
        cmb_beam_fwhm = self.cmb_beam_fwhm.to(u.rad).value
        cmb_beam = hp.gauss_beam(cmb_beam_fwhm, lmax=obs_lmax)
        cmb_pixwin = hp.pixwin(cmb_nside, lmax=obs_lmax, pol=False)

        # Sky prep
        sky_nside = self.sky_nside
        sky_lmax = int(lmax_ratio * sky_nside)
        sky_pixwin = hp.pixwin(nside=sky_nside, lmax=sky_lmax)

        sky_beam_fwhm = sky_beam_fwhm.to(u.rad).value
        sky_beam = hp.gauss_beam(sky_beam_fwhm, obs_lmax)

        # Process CMB map: remove CMB beam and pixwin, apply sky beam and obs pixwin
        cmb_fl = (sky_beam * obs_pixwin) / (cmb_beam * cmb_pixwin)
        cmb_alms = pysm3.map2alm(input_map=cmb_map,
                                 nside=None,
                                 lmax=cmb_lmax,
                                 map2alm_lsq_maxiter=alm_max_iter)
        cmb_alms_bmd = hp.almxfl(cmb_alms, cmb_fl)
        cmb_map_bmd = hp.alm2map(alms=cmb_alms_bmd,
                                 nside=obs_nside)

        # Process obs map: convolve to sky beam
        obs_fl = obs_beam / (obs_beam**2 + beam_eps)  # safe way to handle wide (30GHz) obs beams
        obs_fl = obs_fl * sky_beam
        obs_alms = pysm3.map2alm(input_map=obs_map,
                                 nside=None,
                                 lmax=obs_lmax,
                                 map2alm_lsq_maxiter=alm_max_iter)
        obs_alms_bmd = hp.almxfl(obs_alms, obs_fl)
        obs_map_bmd = hp.alm2map(alms=obs_alms_bmd,
                                 nside=obs_nside)
        
        diff_map_hr = obs_map_bmd - cmb_map_bmd  # hr = high resolution (obs not sky)

        diff_fl = sky_pixwin / obs_pixwin[:sky_lmax+1]

        diff_alms = pysm3.map2alm(input_map=diff_map_hr,
                                  nside=None,
                                  lmax=sky_lmax,
                                  map2alm_lsq_maxiter=alm_max_iter)
        diff_alms_bmd = hp.almxfl(diff_alms, diff_fl)
        diff_map = hp.alm2map(alms=diff_alms_bmd,
                              nside=sky_nside)
        diff_map = u.Quantity(diff_map, self.sky_unit)
        return diff_map

    def process_freq_highres_sky(self, freq: int):
        lmax_ratio = self.lmax_ratio
        alm_max_iter = self.alm_max_iter
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
        sky_beam       = hp.gauss_beam(sky_beam_fwhm, lmax=sky_lmax)
        sky_pixwin     = hp.pixwin(nside=self.sky_nside, lmax=sky_lmax, pol=False)

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
