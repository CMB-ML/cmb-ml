import logging

import hydra
from tqdm import tqdm
from omegaconf import DictConfig

import numpy as np
from sklearn.decomposition import PCA
import pysm3.units as u
import healpy as hp

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from get_data.utils.get_planck_data_ext import get_planck_noise_fn
from cmbml.utils.physics_mask import simple_galactic_mask
from cmbml.utils.physics_ps import get_autopower
from cmbml.utils.physics_downgrade_by_alm import downgrade_by_alm

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler


logger = logging.getLogger(__name__)


class MakePlanckNoiseModelExecutor(BaseStageExecutor):
    """
    MakePlanckAverageNoiseExecutor averages Planck noise simulation maps.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_noise_models')

        self.out_model: Asset = self.assets_out['noise_model']
        
        self.in_sims: Asset = self.assets_in['noise_sims']
        self.in_sims_avg: Asset = self.assets_in['noise_avg']
        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        # For reference:
        in_sim: HealpyMap
        in_det_table_handler: QTableHandler

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        # Check map fields - currently, this only supports Temperature analysis
        map_fields = cfg.scenario.map_fields
        if len(map_fields) != 1:
            raise NotImplementedError("Only single field maps are supported for now.")

        self.output_units = u.Unit(cfg.scenario.units)

        noise_cfg = cfg.model.sim.noise

        self.lmax_ratio = noise_cfg.lmax_ratio_planck_noise
        self.n_sims = noise_cfg.n_planck_noise_sims
        self.nside_lookup = noise_cfg.src_nside_lookup

        self.mask_galactic_size = noise_cfg.mask_galactic_size
        self.mask_galactic_smooth = noise_cfg.mask_galactic_smooth
        self.save_512_avg_for_reviewers = noise_cfg.get('save_512_avg_for_reviewers', False)

        # Created in execute to not bog down initial checks:
        self.masks = {}

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        self.make_masks()  # Created in execute to not bog down initial checks
        for freq, det in self.instrument.dets.items():
            context = dict(freq=freq, fields=det.fields, n_sims=self.n_sims)
            with self.name_tracker.set_contexts(context):
                self.make_noise_model(freq, det)

    def make_masks(self):
        """
        Create masks needed for removing the galactic plane. This only affects power
        spectra.

        This information is contained in the average map, which will be re-added when
        creating the noise maps.
        """
        logger.info("Creating simple galactic plane masks for noise model.")
        # Planck noise sims may be either 1024 or 2048 nside, depending on the detector.
        freqs_1024 = set([30, 44, 70])
        freqs_2048 = set([100, 143, 217, 353, 545, 857])
        use_freqs  = set(self.instrument.dets.keys())
        if self.save_512_avg_for_reviewers:
            self.masks[512] = simple_galactic_mask(nside=512, 
                                                   width=self.mask_galactic_size, 
                                                   smoothing=self.mask_galactic_smooth)
        else:
            if use_freqs & freqs_1024:
                self.masks[1024] = simple_galactic_mask(nside=1024, 
                                                        width=self.mask_galactic_size, 
                                                        smoothing=self.mask_galactic_smooth)
            if use_freqs & freqs_2048:
                self.masks[2048] = simple_galactic_mask(nside=2048, 
                                                        width=self.mask_galactic_size, 
                                                        smoothing=self.mask_galactic_smooth)

    def make_noise_model(self, freq, det):
        noise_ls, noise_ms, noise_map_unit = self.parse_sims(freq, det)
        mean_ps, components, variance = self.make_pca(noise_ls)
        maps_mean, maps_sd = self.make_mean_sd(noise_ms)

        out_fn = self.out_model.path
        logger.debug(f"Saving noise model to {out_fn}")
        np.savez(out_fn,
                 mean_ps=mean_ps,
                 components=components,
                 variance=variance,
                 maps_mean=maps_mean,
                 maps_sd=maps_sd,
                 # PySM3 units to_string includes spaces, parsing from string doesn't :()
                 maps_unit=noise_map_unit.to_string().replace(' ', ''))

    def parse_sims(self, freq, det):
        """
        For each of the Planck noise sims, get the power spectrum, the mean, and the units
        """
        if self.save_512_avg_for_reviewers:
            nside = 512
        else:
            nside = self.nside_lookup[freq]
        n_fields = len(det.fields)
        use_mask = self.masks[nside]
        lmax = int(self.lmax_ratio * nside)

        if n_fields > 1:
            raise NotImplementedError("Only single field maps are supported for now.")

        avg_noise_map = self.in_sims_avg.read(map_field_strs=det.fields)
        noise_ls = []
        noise_ms = []
        with tqdm(total=self.n_sims, 
                    desc=f"Getting PS for {freq} GHz Maps", 
                    position=0,
                    dynamic_ncols=True
                    ) as outer_bar:
            for sim_num in range(self.n_sims):
                # Get noise map data
                fn = get_planck_noise_fn(freq, sim_num)
                with self.name_tracker.set_context('filename', fn):
                    noise_map = self.in_sims.read(map_field_strs=det.fields)

                noise_map = noise_map.to(self.output_units, equivalencies=u.cmb_equivalencies(det.cen_freq))

                # TODO: Remove this section after review; prefer to save at full resolution.
                #       Look into this first though. I thought it would be longer, ~45s per map to downgrade,
                #       but it's only ~5s. This will make noise model generation take longer, but may be faster
                #       when creating the simulations. Downside: need to get autopower beyond bandwidth limit. HM. 
                if self.save_512_avg_for_reviewers:
                    noise_map = downgrade_by_alm(noise_map, target_nside=512)

                # This is the slow part
                noise_map = noise_map - avg_noise_map
                noise_l = get_autopower(noise_map, use_mask, lmax)
                noise_ls.append(noise_l)

                if sim_num == 0:
                    noise_map_unit = noise_map.unit

                # Mean of a map with an apodized mask:
                noise_mean = np.sum(noise_map * use_mask) / np.sum(use_mask)
                noise_ms.append(noise_mean)

                outer_bar.update(1)

        return noise_ls, noise_ms, noise_map_unit

    @staticmethod
    def make_pca(src_cls):
        log_src_cls = np.log10(src_cls)

        pca = PCA().fit(log_src_cls)
        mean_ps = pca.mean_
        components = pca.components_
        variance = pca.explained_variance_
        return mean_ps, components, variance

    @staticmethod
    def make_mean_sd(src_maps_means):
        t = np.array([x.value for x in src_maps_means])
        maps_mean = np.mean(t) * src_maps_means[0].unit
        maps_sd = np.std(t) * src_maps_means[0].unit
        return maps_mean, maps_sd
