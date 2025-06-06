from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)


logger = logging.getLogger(__name__)


class PandasCsvHandler(GenericHandler):
    def read(self, path: Union[Path, str]):
        # Step 1: Read entire CSV without parsing headers
        df_raw = pd.read_csv(path, header=None)

        # Step 2: Assume first row is header, and infer numeric columns from row 2+
        data_portion = df_raw.iloc[2:]
        is_numeric_col = data_portion.apply(
            lambda col: pd.to_numeric(col, errors='coerce').notna().all()
        )

        # Step 3: Look at row 1 (index=1), and check for strings in numeric columns
        row1 = df_raw.iloc[1]
        suspicious = [
            not self._is_numeric(row1[i]) for i, is_numeric in enumerate(is_numeric_col) if is_numeric
        ]

        # Step 4: If 2+ numeric columns contain non-numeric values, assume multi-header
        if sum(suspicious) >= 2:
            df = pd.read_csv(path, header=[0, 1])
            # Remove 'Unnamed' columns
            df.columns = [
                (a, '' if isinstance(b, str) and b.startswith('Unnamed') else b)
                for a, b in df.columns
            ]
        else:
            df = pd.read_csv(path)

        return df

    def _is_numeric(self, val):
        import pandas as pd
        if pd.isna(val) or val == '':
            return True  # Treat blank/missing as safe
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def write(self, 
              path: Union[Path, str], 
              data: Union[dict, pd.DataFrame],
              index: bool=False
              ):
        make_directories(path)
        try:
            data.to_csv(path, index=index)
        except:
            pd.DataFrame(data).to_csv(path, index=index)


register_handler("PandasCsvHandler", PandasCsvHandler)
