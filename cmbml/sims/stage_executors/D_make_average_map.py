import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import pysm3.units as u
import healpy as hp

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from get_data.utils.get_planck_data_ext import get_planck_noise_fn

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler
from cmbml.utils.physics_downgrade_by_alm import downgrade_by_alm


logger = logging.getLogger(__name__)


class MakePlanckAverageNoiseExecutor(BaseStageExecutor):
    """
    MakePlanckAverageNoiseExecutor averages Planck noise simulation maps.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_planck_noise_sims_avgs')

        self.out_avg_sim: Asset = self.assets_out['noise_avg']
        
        self.in_sims: Asset = self.assets_in['noise_sims']
        in_det_table: Asset = self.assets_in['deltabandpass']
        # For reference:
        in_noise_sim: HealpyMap
        in_det_table_handler: QTableHandler

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.output_units = u.Unit(cfg.scenario.units)

        noise_cfg = cfg.model.sim.noise
        self.n_sims = noise_cfg.n_planck_noise_sims
        self.nside_lookup = noise_cfg.src_nside_lookup
        # If the setting exists. Remove after review; prefer to save at full resolution.
        self.save_512_avg_for_reviewers = noise_cfg.get('save_512_avg_for_reviewers', False)

    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        for freq, det in self.instrument.dets.items():
            context = dict(freq=freq, fields=det.fields, n_sims=self.n_sims)
            with self.name_tracker.set_contexts(context):
                self.make_avg_map(freq, det)

    def make_avg_map(self, freq, det):
        nside = self.nside_lookup[freq]
        n_fields = len(det.fields)

        if n_fields == 1:
            out_shape = (1, hp.nside2npix(nside))
        elif n_fields == 3:
            out_shape = (3, hp.nside2npix(nside))
        else:
            raise ValueError(f"Unknown number of fields: {n_fields} for detector: {freq}")

        avg_noise_map = np.zeros(out_shape)
        with tqdm(total=self.n_sims, 
                    desc=f"Averaging {freq} GHz Maps", 
                    position=0,
                    dynamic_ncols=True
                    ) as outer_bar:
            for sim_num in range(self.n_sims):
                # Get noise map data
                fn = get_planck_noise_fn(freq, sim_num)
                with self.name_tracker.set_context('filename', fn):
                    noise_map = self.in_sims.read(map_field_strs=det.fields)

                noise_map = noise_map.to(self.output_units, equivalencies=u.cmb_equivalencies(det.cen_freq))

                # Set units for the average map if the first map has units
                if sim_num == 0:
                    try:
                        avg_noise_map = u.Quantity(avg_noise_map, unit=self.output_units)
                    except AttributeError:
                        logger.warning(f"No units found for {freq} map!")

                # Update average
                avg_noise_map += noise_map / self.n_sims
                outer_bar.update(1)

        if self.save_512_avg_for_reviewers:
            avg_noise_map = downgrade_by_alm(avg_noise_map, target_nside=512)

        # Prepare FITS header information & save maps
        column_names = [f"STOKES_{x}" for x in det.fields]
        extra_header = [("METHOD", f"FROM_SIMS", f"Average of {self.n_sims} Planck 2018 noise simulations")]
        self.out_avg_sim.write(data=avg_noise_map, 
                               column_names=column_names,
                               extra_header=extra_header)

        logger.debug(f"Averaging complete for {freq}")
