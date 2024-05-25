from typing import Dict, Union
import logging
from pathlib import Path

import torch

from src.core.asset_handlers import GenericHandler
from src.core.asset_handlers import make_directories
from src.core.asset import register_handler


logger = logging.getLogger(__name__)


class PyTorchModel(GenericHandler):
    def read(self, path: Path, 
             model: torch.nn.Module, 
             epoch: str, 
             optimizer=None, 
             scheduler=None) -> Dict:
        logger.debug(f"Reading model from '{path}'")
        fn_template = path.name
        fn = fn_template.format(epoch=epoch)
        this_path = path.parent / fn
        checkpoint = torch.load(this_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch']

    def write(self, 
              path: Path, 
              model: torch.nn.Module, 
              epoch: Union[int, str], 
              optimizer = None,
              scheduler = None,
              loss = None,
              ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if loss is not None:
            checkpoint['loss'] = loss

        new_path = Path(str(path).format(epoch=epoch))
        make_directories(new_path)
        logger.debug(f"Writing model to '{new_path}'")
        torch.save(checkpoint, new_path)


register_handler("PyTorchModel", PyTorchModel)
