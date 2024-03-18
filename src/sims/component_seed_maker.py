from typing import List
from hashlib import sha256
from omegaconf.errors import ConfigAttributeError
from omegaconf import DictConfig
import logging

from src.sims.namer_dataset_output import SplitFilesNamer, SimFilesNamer


logger = logging.getLogger("seed_logger")

class SeedMaker:
    def __init__(self, 
                 conf: DictConfig, 
                 sky_component: str, 
                 use_backup_strs: bool=False) -> None:
        self.base: str = self.get_base_string(conf)
        self.component: str = self.get_component_string(conf, sky_component)
        self.use_backup_strs = use_backup_strs
        try:
            "".join([self.base, ""])
        except TypeError as e:
            raise e

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
                 split: SplitFilesNamer, 
                 sim: SimFilesNamer) -> int:
        split_str = split.name
        sim_str = sim.sim_num_str
        return self._get_seed(split_str, sim_str, self.component)


class FieldLevelSeedFactory(SeedMaker):
    def __init__(self, 
                 conf: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(conf, sky_component)

    def get_seed(self, 
                 split: SplitFilesNamer, 
                 sim: SimFilesNamer, 
                 freq: int, 
                 field_str: str):
        split_str = split.name
        sim_str = sim.sim_num_str
        freq_str = str(freq)
        return self._get_seed(split_str, sim_str, freq_str, field_str, self.component)
