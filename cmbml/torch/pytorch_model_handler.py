from typing import Dict, Union, Optional, Any
import logging
from pathlib import Path

import torch

from ..core.asset_handlers.asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)


logger = logging.getLogger(__name__)


class PyTorchModel(GenericHandler):
    def read(self, 
             path: Path, 
             model: torch.nn.Module, 
             epoch: Optional[Union[int, str]] = None, 
             optimizer=None, 
             scheduler=None,
             scaler=None,
             strict: bool=True,
             map_location: Union[str, torch.device] = "cpu"
             ) -> Dict[str, Any]:
        logger.debug(f"Reading model from '{path}'")
        fn_template = path.name
        if epoch is None and "{epoch}" in fn_template:
            raise ValueError("Path template expects 'epoch', but epoch=None was provided.")
        fn = fn_template if epoch is None else fn_template.format(epoch=epoch)
        this_path = path.parent / fn
        checkpoint = torch.load(this_path,
                                map_location=map_location,
                                weights_only=True)
        
        incompat = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict
        )

        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'model_config' in checkpoint:
            model.model_config = checkpoint['model_config']

        return {
            "epoch": checkpoint.get("epoch"),
            "best_loss": checkpoint.get("best_loss", None),
            "incompat": incompat,
        }

    def write(self, 
              path: Path, 
              model: torch.nn.Module, 
              epoch: Union[int, str], 
              optimizer = None,
              scheduler = None,
              scaler = None,
              best_loss = None,
              ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss
        if hasattr(model, "model_config"):
            checkpoint['model_config'] = model.model_config

        new_path = Path(str(path).format(epoch=epoch))
        make_directories(new_path)
        logger.debug(f"Writing model to '{new_path}'")
        torch.save(checkpoint, new_path)


register_handler("PyTorchModel", PyTorchModel)
