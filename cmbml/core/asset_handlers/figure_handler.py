import logging
from pathlib import Path
from typing import Union
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
# import shutil

from .asset_handlers_base import GenericHandler, make_directories
from .asset_handler_registration import register_handler


logger = logging.getLogger(__name__)


class Figure(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path) -> Path:
        logger.debug(f"Creating parent directory at {path}")
        make_directories(path)
        return path


class MPLFigure(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path, fig, verbose=False, **kwargs) -> Path:
        logger.debug(f"Creating parent directory at {path}")
        make_directories(path)

        fig_types = getattr(self, "fig_types", None)
        if not fig_types:
            raise ValueError(
                f"No figure formats configured for {path}. "
                "Call set_fig_types() with a valid format."
            )

        log = logger.info if verbose else logger.debug
        log(f"Saving figure to {path.with_suffix('')} as {fig_types}")

        for fig_type in fig_types:
            fig.savefig(path.with_suffix(f".{fig_type.lstrip('.')}"), **kwargs)
        plt.close(fig)

    def set_fig_types(self, fig_types):
        if OmegaConf.is_config(fig_types):
            fig_types = OmegaConf.to_container(fig_types)

        if not isinstance(fig_types, (list, tuple)):
            fig_types = [fig_types]

        if not fig_types or any(
            not isinstance(fmt, str) or not fmt.strip(". ")
            for fmt in fig_types
        ):
            raise ValueError(
                f"Invalid fig_types: {fig_types!r}. "
                "Expected one or more figure formats, such as 'png' or 'pdf'."
            )

        self.fig_types = [fmt.strip().lstrip(".") for fmt in fig_types]

register_handler("Figure", Figure)
register_handler("MPLFigure", MPLFigure)
