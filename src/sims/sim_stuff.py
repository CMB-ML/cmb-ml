
from typing import List, Dict, Any
import logging
from hashlib import sha256
from omegaconf import DictConfig
from pathlib import Path
from pysm3 import CMBLensed

from omegaconf.errors import ConfigAttributeError

from ..core import Split, Asset

logger = logging.getLogger(__name__)

class CMBFactory:
    def __init__(self, conf, make_ps_if_absent=None):
        self.nside = conf.simulation.nside_sky
        self.max_ell_for_camb = conf.simulation.cmb.ell_max
        self.wmap_param_labels = conf.simulation.cmb.wmap_params
        self.camb_param_labels = conf.simulation.cmb.camb_params_equiv
        self.max_nside_pysm_component = None
        self.apply_delens = False
        self.delensing_ells = None
        self.map_dist = None
        if make_ps_if_absent is None:
            try:
                self.make_ps_if_absent = conf.simulation.cmb.make_ps_if_absent
            except ConfigAttributeError as e:
                logger.error(f"CMBMaker needs either a make_ps_if_absent flag in the cmb " \
                             f"configuration yaml OR a make_ps_if absent argument to init.")
                logger.exception(e)
                raise e
    
    def make_cmb_lensed(self, seed, sim_files: Asset) -> CMBLensed:
        logger.debug(f'Seed used for CMB: {seed}.')
        cmb_ps_fid_path = sim_files.path
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)

class SeedMaker:
    def __init__(self, 
                 conf: DictConfig, 
                 sky_component: str, 
                 use_backup_strs: bool=False) -> None:
        self.base: str = self.get_base_string(conf)
        self.component: str = self.get_component_string(conf, sky_component)
        self.use_backup_strs = use_backup_strs
        self.str_num_digits = conf.file_system.sim_str_num_digits
        try:
            "".join([self.base, ""])
        except TypeError as e:
            raise e

    def sim_num_str(self, sim: int) -> str:
        return f"{sim:0{self.str_num_digits}d}"
    
    def get_base_string(self, 
                        conf: DictConfig):
        try:
            base_string = conf.seed_base_string
        except ConfigAttributeError as e:
            if self.use_backup_strs:
                logger.info(f"No seed string set for seed_base_string in yaml; using conf.dataset_name:'{conf.dataset_name}'")
                base_string = conf.dataset_name
            else:
                logger.error(f"No seed string set for seed_base_string in yaml; backup string disabled.")
                raise e
        return str(base_string)

    def get_component_string(self, 
                             conf: DictConfig, 
                             sky_component: str) -> str:
        try:
            base_string = conf.simulation[sky_component].seed_string
            pass
        except ConfigAttributeError as e:
            if self.use_backup_strs:
                logger.warning(f"No seed string set for {sky_component} yaml; using '{sky_component}'.")
                base_string = sky_component
            else:
                logger.error(f"No seed string set for {sky_component} yaml; backup string disabled.")
                raise e
        return str(base_string)

    def _get_seed(self, *args: List[str]) -> int:
        try:
            str_list = [self.base, *args]
            seed_str = "_".join(str_list)
        except Exception as e:
            raise e
        seed_int = self.string_to_seed(seed_str)
        return seed_int

    @staticmethod
    def string_to_seed(input_string: str) -> int:
        hash_object = sha256(input_string.encode())
        # Convert the hash to an integer
        hash_integer = int(hash_object.hexdigest(), 16)
        # Reduce the size to fit into expected seed range of ints (for numpy/pysm3)
        seed = hash_integer % (2**32)
        logger.info(f"Seed for {input_string} is {seed}.")
        return seed


class SimLevelSeedFactory(SeedMaker):
    def __init__(self, 
                 conf: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(conf, sky_component)

    def get_seed(self, 
                 split: Split, 
                 sim: int) -> int:
        split_str = split.name
        sim_str = self.sim_num_str(sim)
        return self._get_seed(split_str, sim_str, self.component)


class FieldLevelSeedFactory(SeedMaker):
    def __init__(self, 
                 conf: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(conf, sky_component)

    def get_seed(self, 
                 split: Split, 
                 sim: int, 
                 freq: int, 
                 field_str: str):
        split_str = split.name
        sim_str = self.sim_num_str(sim)
        freq_str = str(freq)
        return self._get_seed(split_str, sim_str, freq_str, field_str, self.component)
