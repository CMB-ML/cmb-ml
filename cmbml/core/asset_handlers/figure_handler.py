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

    def write(self, path: Path, fig, **kwargs) -> Path:
        logger.debug(f"Creating parent directory at {path}")
        make_directories(path)

        fig_types = getattr(self, "fig_types", None)
        if not fig_types:
            logger.warning(
                f"No fig_types set on {type(self).__name__}; saving only "
                f"{path.suffix or 'the default format'}. Call set_fig_types()."
            )
            fig_types = [path.suffix or "png"]

        for fig_type in fig_types:
            out = path.with_suffix(f".{fig_type.lstrip('.')}")
            logger.debug(f"Saving figure to {out}")
            fig.savefig(out, **kwargs)
        plt.close(fig)

    def set_fig_types(self, fig_types):
        try:
            fig_types = OmegaConf.to_container(fig_types)
        except ValueError:
            pass
        if not isinstance(fig_types, list):
            fig_types = [fig_types]
        self.fig_types = fig_types

register_handler("Figure", Figure)
register_handler("MPLFigure", MPLFigure)
