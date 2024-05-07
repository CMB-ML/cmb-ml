from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path

from astropy.units import Quantity

from ...core import BaseStageExecutor, Asset
from ..planck_instrument import make_instrument, Instrument
from utils.fits_inspection import get_num_fields_in_hdr
from ..physics_instrument_noise import planck_result_to_sd_map

from ...core.asset_handlers import HealpyMap
from ..handlers.qtable_handler import QTableHandler


logger = logging.getLogger(__name__)
class NoiseCacheExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_noise_cache')

        self.out_noise_cache: Asset = self.assets_out['noise_cache']
        self.in_noise_src: Asset = self.assets_in['noise_src_maps']
        in_det_table: Asset = self.assets_in['planck_deltabandpass']

        # For reference:
        out_noise_cache_handler: HealpyMap
        in_noise_src_handler: HealpyMap
        in_det_table_handler: QTableHandler

        with self.name_tracker.set_context('src_root', cfg.local_system.noise_src_dir):
            det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.noise_src_dir: str            = cfg.local_system.noise_src_dir

    def execute(self) -> None:
        hdu = self.cfg.simulation.noise.hdu_n
        nside = self.cfg.scenario.nside
        for freq, detector in self.instrument.dets.items():
            src_path = self.get_src_path(freq)
            for field_str in detector.fields:
                field_idx = self.get_field_idx(src_path, field_str)
                st_dev_skymap = planck_result_to_sd_map(fits_fn=src_path, 
                                                        hdu=hdu, 
                                                        field_idx=field_idx, 
                                                        nside_out=nside, 
                                                        cen_freq=detector.cen_freq)
                with self.name_tracker.set_contexts(dict(freq=freq, field=field_str)):
                    self.write_wrapper(data=st_dev_skymap, field_str=field_str)

    def write_wrapper(self, data: Quantity, field_str):
        units = data.unit
        data = data.value
        
        # We want to give some indication that for I field, this is from the II covariance (or QQ, UU)
        col_name = field_str + field_str
        logger.debug(f'Writing NoiseCache map to path: {self.out_noise_cache.path}')
        self.out_noise_cache.write(data=data,
                                   column_names=[col_name],
                                   column_units=[units]
                                   )
        logger.debug(f'Wrote NoiseCache map to path: {self.out_noise_cache.path}')

    def get_field_idx(self, src_path, field_str) -> int:
        """
        Looks at fits file to determine field_idx corresponding to field_str

        Parameters:
        src_path (str): The path to the fits file.
        field_str (str): The field string to look up.

        Returns:
        int: The field index corresponding to the field string.
        """
        hdu = self.cfg.simulation.noise.hdu_n
        field_idcs_dict = dict(self.cfg.simulation.noise.field_idcs)
        # Get number of fields in map
        n_map_fields = get_num_fields_in_hdr(fits_fn=src_path, hdu=hdu)
        # Lookup field index based on config file
        field_idx = field_idcs_dict[n_map_fields][field_str]
        return field_idx


    def get_src_path(self, detector: int):
        """
        Get the path for the source noise file based on the hydra configs.

        Parameters:
        detector (int): The nominal frequency of the detector.

        Returns:
        str: The path for the fits file containing the noise.
        """
        fn = self.cfg.simulation.noise.src_files[detector]
        noise_src_dir = self.cfg.local_system.noise_src_dir
        contexts_dict = dict(src_root=noise_src_dir, fn=fn)
        with self.name_tracker.set_contexts(contexts_dict):
            src_path = self.in_noise_src.path
        return src_path
