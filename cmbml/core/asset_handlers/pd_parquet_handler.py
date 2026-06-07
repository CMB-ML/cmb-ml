from typing import Union
from pathlib import Path
import logging

import pandas as pd

from .asset_handlers_base import (
    GenericHandler,
    register_handler,
    make_directories,
)


logger = logging.getLogger(__name__)


class PandasParquetHandler(GenericHandler):
    def read(
        self,
        path: Union[Path, str],
        engine: str = "pyarrow",
        columns=None,
    ):
        return pd.read_parquet(
            path,
            engine=engine,
            columns=columns,
        )

    def write(
        self,
        path: Union[Path, str],
        data: Union[dict, pd.DataFrame],
        engine: str = "pyarrow",
        compression: str = "zstd",
        index: bool = False,
    ):
        make_directories(path)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data.to_parquet(
            path,
            index=index,
            engine=engine,
            compression=compression,
        )


register_handler("PandasParquetHandler", PandasParquetHandler)
