
from typing import List
import logging
from hashlib import sha256
from omegaconf import DictConfig

from omegaconf.errors import ConfigAttributeError

from cmbml.core import Split


logger = logging.getLogger('seed_logger')


class SeedMaker:
    """
    A class to generate seeds for simulations. Uses the SHA-256 hash of
    a string composed of the string representing the split, the simulation
    number, the component string, and a base string.

    Attributes:
        base (str): The base string for the seed.
        component (str): The component string for the seed.
        str_num_digits (int): The number of digits in the simulation string.

    Methods:
        sim_num_str: Return a string representation of a simulation number.
        get_base_string: Get the base string for the seed.
        get_component_string: Get the component string for the seed.
        string_to_seed: Convert a string to a seed.
    """
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
        """
        Convert a simulation number to a string with a fixed number of digits.

        Args:
            sim (int): The simulation number.

        Returns:
            str: The simulation number as a string.
        """
        return f"{sim:0{self.str_num_digits}d}"
    
    def get_base_string(self, 
                        cfg: DictConfig):
        """
        Get the base string for the seed from the configuration.

        Args:
            cfg (DictConfig): The configuration object.

        Returns:
            str: The base string.
        """
        base_string = cfg.model.sim.seed_base_string
        return str(base_string)

    def get_component_string(self, 
                             cfg: DictConfig, 
                             sky_component: str) -> str:
        """
        Get the component string for the seed from the configuration.

        Args:
            cfg (DictConfig): The configuration object.
            sky_component (str): The sky component.

        Returns:
            str: The component string
        """
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
        """
        Convert a string to a seed using the SHA-256 hash.
        
        Args:
            input_string (str): The input string.

        Returns:
            int: The seed.
        """
        hash_object = sha256(input_string.encode())
        # Convert the hash to an integer
        hash_integer = int(hash_object.hexdigest(), 16)
        # Reduce the size to fit into expected seed range of ints (for numpy/pysm3)
        seed = hash_integer % (2**32)
        logger.info(f"Seed for {input_string} is {seed}.")
        return seed


class SimLevelSeedFactory(SeedMaker):
    """
    Some components of the sky model are the same for all fields.
    This class generates seeds for these components.

    Attributes:
        cfg (DictConfig): The Hydra config to use.
        sky_component (str): The sky component to use.

    Methods:
        get_seed(split, sim): Generate and retrieve a seed.
    """
    def __init__(self, 
                 cfg: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(cfg, sky_component)

    def get_seed(self, 
                 split: Split, 
                 sim: int) -> int:
        """
        Generate and retrieve the seed of the
        specified split for a simulation.

        Args:
            split (Split): The specified split.
            sim (int): The specified simulation.

        Returns:
            int: The generated seed.
        """
        split_str = split.name
        sim_str = self.sim_num_str(sim)
        return self._get_seed(split_str, sim_str, self.component)


class FieldLevelSeedFactory(SeedMaker):
    """
    Some components of the sky model need seeds for each field.
    This class generates seeds for these components.

    Attributes:
        cfg (DictConfig): The Hydra config to use.
        sky_component (str): The sky component to use.

    Methods:
        get_seed(split, sim): Generate and retrieve a seed.
    """
    def __init__(self, 
                 cfg: DictConfig, 
                 sky_component: str) -> None:
        super().__init__(cfg, sky_component)

    def get_seed(self, 
                 split: str, 
                 sim: int, 
                 freq: int, 
                 field_str: str):
        """
        Generate and retrieve the seed of the
        specified field.

        Args:
            split (Split): The specified split.
            sim (int): The specified simulation.
            freq (int): The specified frequency.
            field_str (str): The specified field string.

        Returns:
            int: The generated seed.
        """
        split_str = split
        sim_str = self.sim_num_str(sim)
        freq_str = str(freq)
        return self._get_seed(split_str, sim_str, freq_str, field_str, self.component)
