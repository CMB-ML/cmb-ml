from typing import Any, Dict, List, Union
from pathlib import Path
import logging
import csv

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)


logger = logging.getLogger(__name__)


class AppendingCsvHandler(GenericHandler):
    def read(self, path: Union[Path, str]):
        raise NotImplementedError("This method is not implemented yet.")

    def start(self,
              path: Union[Path, str], 
              newline: str=''
              ) -> None:
        self.destination = path
        self.newline = newline

    def write(self, 
              path: Union[Path, str], 
              data: Union[tuple, list],
              newline: str=''
              ) -> None:
        self.start(path, newline)
        make_directories(path)

        if path.exists():
            logger.warning(f"Overwriting existing file: {path}")
            path.unlink()
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def append(self,
               data: Union[tuple, list]) -> None:
        if self.destination is None:
            raise ValueError("No destination file specified. Use write() first.")
        with open(self.destination, 'a', newline=self.newline) as f:
            writer = csv.writer(f)
            writer.writerow(data)


register_handler("AppendingCsvHandler", AppendingCsvHandler)
