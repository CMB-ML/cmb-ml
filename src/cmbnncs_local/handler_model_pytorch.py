from typing import Dict, Union
import logging
from pathlib import Path

import torch

from core import GenericHandler
from core import register_handler
from core.asset_handlers import make_directories


logger = logging.getLogger(__name__)


class PyTorchModel(GenericHandler):
    def read(self, path: Path, model: torch.nn.Module) -> Dict:
        # fn_template = path.name
        # fn = fn_template.format(epoch=epoch)
        logger.debug(f"Reading model from '{path}'")
        # this_path = path.parent / fn
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def write(self, 
              path: Path, 
              model: torch.nn.Module, 
              epoch: Union[int, str], 
              optimizer = None,
              loss = None,
              ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if loss is not None:
            checkpoint['loss'] = loss

        # path = Path(str(path).format(epoch=epoch))
        make_directories(path)
        logger.debug(f"Writing model to '{path}'")
        torch.save(checkpoint, path)


register_handler("PyTorchModel", PyTorchModel)
