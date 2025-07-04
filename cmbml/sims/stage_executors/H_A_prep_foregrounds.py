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

        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        in_noise_cache_handler: Union[HealpyMap, NumpyPowerSpectrum]
        in_cmb_ps_handler: CambPowerSpectrum
        in_det_table_handler: QTableHandler

        # Initialize constants from configs
        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)
        
        self.src_files_dict = cfg.model.sim.noise.src_files
        self.assets_dir     = cfg.local_system.assets_dir
        self.cmb_map        = None
        self.unit           = u.K_RJ  # Matching PySM3 default
        self.cmb_beam_fwhm  = 5 * u.arcmin
        self.lmax_ratio     = 2.5
        self.nside_sky      = cfg.model.sim.nside_sky
        self.alm_max_iter   = 50

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method")

        self.check_files_exist()

        for freq, det in self.instrument.dets.items():
            fn = self.src_files_dict[freq]
            context = dict(freq=freq, filename=fn)
            with self.name_tracker.set_contexts(context):
                self.process_freq(det)
            logger.info(f"Processed frequency {freq} for foregrounds")

    def check_files_exist(self):
        if not self.in_cmb_map.path.exists():
            raise FileNotFoundError(f"CMB map file not found: {self.in_cmb_map.path}")

        for freq in self.instrument.dets.keys():
            fn = self.src_files_dict[freq]
            context = dict(freq=freq, filename=fn)
            with self.name_tracker.set_contexts(context):
                obs_path = self.in_obs_maps.path
                if not obs_path.exists():
                    raise FileNotFoundError(f"Observation map file not found for frequency {freq}: {obs_path}")

    def process_freq(self, det: Detector):
        freq = det.nom_freq
        # We're doing temperature-only for now.
        obs_map = self.in_obs_maps.read()[0]
        cmb_map = self.in_cmb_map.read()[0]

        obs_nside = hp.get_nside(obs_map)
        obs_lmax = int(self.lmax_ratio * obs_nside)

        cmb_nside = hp.get_nside(cmb_map)

        logger.info(f"Getting Obs alms ({freq} GHz)")
        obs_alms = self._process_and_convert_map_to_alms(obs_map, det, lmax=obs_lmax)

        logger.info(f"Getting CMB alms")
        # Use obs_lmax; don't need more alms than that
        cmb_alms = self._process_and_convert_map_to_alms(cmb_map, det, lmax=obs_lmax)

        # cmb_lmax = int(self.lmax_ratio * cmb_nside)

        obs_beam_fwhm = det.fwhm.to(u.rad).value
        obs_beam       = hp.gauss_beam(obs_beam_fwhm, lmax=obs_lmax)
        obs_pixwin     = hp.pixwin(nside=obs_nside, lmax=obs_lmax, pol=False)

        cmb_beam_fwhm = self.cmb_beam_fwhm.to(u.rad).value
        cmb_beam = hp.gauss_beam(cmb_beam_fwhm, lmax=obs_lmax)
        cmb_pixwin = hp.pixwin(nside=cmb_nside, lmax=obs_lmax, pol=False)

        to_obs_fl = (obs_beam * obs_pixwin) / (cmb_beam * cmb_pixwin)
        # cmb_alms = hp.resize_alm(cmb_alms, 
        #                          lmax=cmb_lmax,
        #                          mmax=cmb_lmax,
        #                          lmax_out=obs_lmax,
        #                          mmax_out=obs_lmax)
        cmb_alms = hp.almxfl(cmb_alms, to_obs_fl)

        diff_alm = obs_alms - cmb_alms

        # debeam_fl = 1 / (obs_beam * obs_pixwin)
        debeam_fl_safe = (obs_beam * obs_pixwin) / ((obs_beam * obs_pixwin)**2 + 1e-10)
        diff_debeam_alm = hp.almxfl(diff_alm, debeam_fl_safe)

        diff_map = hp.alm2map(alms=diff_debeam_alm,
                              nside=self.nside_sky, 
                              pixwin=True)
        
        diff_map *= self.unit
        col_names = ["I_FG_ONLY"]
        self.out_foregrounds.write(data=diff_map, column_names=col_names)

    def _process_and_convert_map_to_alms(self, map_data, detector, lmax):
        map_data = inpaint_with_neighbor_mean(map_data)
        map_data = map_data.to(self.unit,
                               equivalencies=u.cmb_equivalencies(detector.cen_freq))
        # nside = hp.get_nside(map_data)
        # lmax = int(self.lmax_ratio * nside)
        alm = pysm3.map2alm(map_data, nside=None, lmax=lmax, 
                            map2alm_lsq_maxiter=self.alm_max_iter)
        return alm
