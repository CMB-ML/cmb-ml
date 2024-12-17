
from typing import Dict
import logging
from hashlib import sha256
from omegaconf import DictConfig

from omegaconf.errors import ConfigAttributeError

from cmbml.core import Split


logger = logging.getLogger('seed_logger')


class SeedFactory:
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
                 seed_template: str) -> None:
        # self.base: str = cfg.model.sim.seed_base_string
        # self.str_num_digits = cfg.file_system.sim_str_num_digits
        self.seed_template: str = seed_template

    # def sim_num_str(self, sim: int) -> str:
    #     """
    #     Convert a simulation number to a string with a fixed number of digits.

    #     Args:
    #         sim (int): The simulation number.

    #     Returns:
    #         str: The simulation number as a string.
    #     """
    #     return f"{sim:0{self.str_num_digits}d}"

    def get_seed(self, **kwargs: Dict[str, str]) -> int:
        """
        Takes an arbitrary number of keyword arguments and returns a seed

        Arguments must be given as keyword arguments (e.g., x=y)!

        Argument keys are as defined in the seed_template string,
        (e.g., in cfg/model/sim/cmb/whatever.yaml, 
               in cfg/model/sim/noise/whatever.yaml, 
               or cfg/model/patches/whatever.yaml)
        under the key "seed_template"
        """
        seed_params = {**kwargs}
        # seed_params["base"] = self.base
        seed_str = self.seed_template.format(**seed_params)
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
