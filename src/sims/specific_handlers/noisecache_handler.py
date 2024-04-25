from pathlib import Path
from typing import Union
import healpy as hp

from ...core.asset_handlers import GenericHandler, _make_directories
from ...core.asset_handler_registration import register_handler


class NoiseCacheHandler(GenericHandler):
    def read(self, path: Union[Path, str]):
        # Check with jim of this is the right way to read in the noise cache
        path = Path(path)
        sd_map = hp.read_map(path)
        return sd_map

    def write(self, path: Union[Path, str], st_dev_skymap, field_str):
        path = Path(path)
        _make_directories(path)
        col_names = {"T": "II", "Q": "QQ", "U": "UU"}
        hp.write_map(filename=str(path),
                     m=st_dev_skymap,
                     nest=False,
                     column_names=[col_names[field_str]],
                     column_units=["K_CMB"],
                     dtype=st_dev_skymap.dtype,
                     overwrite=True
                    # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
                    #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
                     )

register_handler('NoiseCache', NoiseCacheHandler)