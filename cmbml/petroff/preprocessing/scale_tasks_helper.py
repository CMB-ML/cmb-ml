from typing import NamedTuple
from pathlib import Path
from cmbml.core.asset_handlers import GenericHandler


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    cmb_asset: FrozenAsset
    obs_asset: FrozenAsset
    split_name: str
    sim_num: str
