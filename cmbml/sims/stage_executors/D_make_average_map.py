import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import pysm3.units as u
import healpy as hp

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils.get_planck_data import get_planck_noise_fn

from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler


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
        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        # For reference:
        in_noise_sim: HealpyMap
        in_det_table_handler: QTableHandler

        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.n_sims = cfg.model.sim.noise.n_planck_noise_sims
        self.nside_lookup = cfg.model.sim.noise.src_nside_lookup

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
            out_shape = hp.nside2npix(nside)
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

                # Set units for the average map using units in the first map
                if sim_num == 0:
                    try:
                        avg_noise_map = u.Quantity(noise_map, unit=noise_map.unit)
                    except AttributeError:
                        logger.warning(f"No units found for {freq} map!")
                        pass

                # Update average
                avg_noise_map += noise_map / self.n_sims
                outer_bar.update(1)

        # Prepare FITS header information & save maps
        column_names = [f"STOKES_{x}" for x in det.fields]
        extra_header = [("METHOD", f"FROM_SIMS", f"Average of {self.n_sims} Planck 2018 noise simulations")]
        self.out_avg_sim.write(data=avg_noise_map, 
                               column_names=column_names,
                               extra_header=extra_header)

        logger.debug(f"Averaging complete for {freq}")
