from typing import Any, Dict, List, Union
import shutil
from pathlib import Path
import yaml
import logging

import numpy as np
import pysm3.units as u

from .asset_handler_registration import register_handler
from .asset_handlers_base import GenericHandler, make_directories


logger = logging.getLogger(__name__)


class Config(GenericHandler):
    def read(self, path: Path) -> Dict:
        """
        Reads YAML config and converts it to Python objects, including restoring Astropy Quantities.
        """
        try:
            with open(path, 'r') as infile:
                data = yaml.safe_load(infile)
            # Restore Quantities from dict representation
            data = self._restore_quantities(data)
        except Exception as e:
            logger.error(f"Failed to read file at '{path}': {e}")
            raise
        return data

    def write(self, path, data, verbose=True) -> None:
        if verbose:
            logger.debug(f"Writing config to '{path}'")
        make_directories(path)
        un_obj_data = self._convert_obj(data)

        # Patch to handle the yaml library not liking square brackets in entries
        #    addressing the config for input to the PyILC code
        yaml_string = yaml.dump(un_obj_data, default_flow_style=False)
        if "\[" in yaml_string and "\]" in yaml_string:
            yaml_string = yaml_string.replace("\[", "[").replace("\]", "]")
        try:
            with open(path, 'w') as outfile:
                outfile.write(yaml_string)
        except Exception as e:
            logger.error(f"Failed to write file at '{path}': {e}")
            raise

    @staticmethod
    def _convert_obj(obj: Union[Dict[str, Any], List[Any], np.generic]) -> Any:
        # Recursive function
        # The `Any`s in the above signature should be the same as the signature itself
        # GPT4, minimal modification
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, u.Quantity):
            return {'value': Config._convert_obj(obj.value), 'aq_unit': str(obj.unit)}  # Convert Quantity to dict
        elif isinstance(obj, dict):
            return {key: Config._convert_obj(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [Config._convert_obj(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _restore_quantities(obj: Any) -> Any:
        """
        Recursive function to restore Astropy Quantities from their dictionary representation.

        Looks for dictionaries with 'value' and 'aq_unit' keys and converts them to Quantity objects.
        """
        if isinstance(obj, dict) and 'value' in obj and 'aq_unit' in obj:
            try:
                return u.Quantity(obj['value'], u.Unit(obj['aq_unit']))
            except Exception:
                return obj  # Return original if conversion fails
        elif isinstance(obj, dict):
            return {key: Config._restore_quantities(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [Config._restore_quantities(item) for item in obj]
        else:
            return obj


register_handler("Config", Config)
