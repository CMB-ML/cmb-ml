
from typing import List
import logging
from hashlib import sha256
from omegaconf import DictConfig

from omegaconf.errors import ConfigAttributeError

from src.core import Split


logger = logging.getLogger('seed_logger')


class SeedMaker:
    def __init__(self, 
                 cfg: DictConfig, 
                 sky_component: str) -> None:
        self.base: str = self.get_base_string(cfg)
        self.component: str = self.get_component_string(cfg, sky_component)
        self.str_num_digits = cfg.file_system.sim_str_num_digits
        try:
            "".join([self.base, ""])
        except TypeError as e:
            raise e

    def sim_num_str(self, sim: int) -> str:
        return f"{sim:0{self.str_num_digits}d}"
    
    def get_base_string(self, 
                        cfg: DictConfig):
        base_string = cfg.model.sim.seed_base_string
        return str(base_string)

    def get_component_string(self, 
                             cfg: DictConfig, 
                             sky_component: str) -> str:
        try:
            base_string = cfg.model.sim[sky_component].seed_string
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
                 cfg: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(cfg, sky_component)

    def get_seed(self, 
                 split: Split, 
                 sim: int) -> int:
        split_str = split.name
        sim_str = self.sim_num_str(sim)
        return self._get_seed(split_str, sim_str, self.component)


class FieldLevelSeedFactory(SeedMaker):
    def __init__(self, 
                 cfg: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(cfg, sky_component)

    def get_seed(self, 
                 split: str, 
                 sim: int, 
                 freq: int, 
                 field_str: str):
        split_str = split
        sim_str = self.sim_num_str(sim)
        freq_str = str(freq)
        return self._get_seed(split_str, sim_str, freq_str, field_str, self.component)
