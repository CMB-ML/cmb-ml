import hydra

import torch

from core.asset import Asset, AssetWithPathAlts
from core.namers import Namer
from core.config_helper import ConfigHelper
from core.pytorch_make_dataset import create_dataset_from_cfg
from core.pytorch_transform import ToTensorDtype

from petroff.pytorch_transform_absmax_scale import MinMaxScaleMap


def get_scale_factors(cfg, name_tracker):
    stage_str = "preprocess"
    cfg_h = ConfigHelper(cfg, stage_str=stage_str)
    in_norm_file: Asset = cfg_h.get_some_asset_in(name_tracker, "norm_file", stage_str)
    scale_factors = in_norm_file.read()
    return scale_factors


@hydra.main(version_base=None, config_path="cfg", config_name="config_petroff_32")
def main(cfg):
    stage_str = "preprocess"
    split_name = "Train"

    name_tracker = Namer(cfg)
    dataset = create_dataset_from_cfg(cfg, stage_str, split_name, name_tracker)

    dtype = torch.float32
    device = "cpu"

    scale_factors = get_scale_factors(cfg, name_tracker)

    to_tensor = ToTensorDtype(dtype)
    normalize = MinMaxScaleMap("I", scale_factors, dtype=dtype, device=device)

    for i in dataset:
        j = to_tensor(i)
        k = normalize(j)
        print("input", i)
        print("to_tensor", j)
        print("norm", k)
        break

main()
