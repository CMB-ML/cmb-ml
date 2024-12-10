from typing import Union, List, Callable
from utils.planck_instrument import Instrument
from .pytorch_dataset import LabelledCMBMapDataset
from .config_helper import ConfigHelper
from .split import Split
from .namers import Namer
from .asset_handlers.healpy_map_handler import HealpyMap


def create_dataset_from_cfg(cfg, stage_str: str, split_name: str, name_tracker: Namer, 
                            transforms: Union[List[Callable], Callable]=None) -> LabelledCMBMapDataset:
    """
    Get a labelled CMB map dataset from the given configuration.
    Intended for use outside executors.

    Args:
        cfg (str): The configuration file path.
        stage_str (str): The stage string. (To be removed?)
        split_name (str): The name of the split.
        name_tracker: The name tracker object.
        transforms: a single or a list of callable objects that
                    take data as a tuple of (obs, cmb), where
                    each obs and cmb are np.ndarrays or tensors

    Returns:
        LabelledCMBMapDataset: The labelled CMB map dataset.
    """

    # # Get stuff from cfg files
    config_handler         = ConfigHelper(cfg, stage_str)

    instrument: Instrument = config_handler.get_instrument()
    split: Split           = config_handler.get_split(split_name)
    assets_in              = config_handler.get_assets_in(name_tracker)
    map_fields             = config_handler.get_map_fields()

    obs_asset = assets_in["obs_maps"]
    cmb_asset = assets_in["cmb_map"]

    context = dict(
        split=split.name,
        sim=name_tracker.sim_name_template,
        freq="{freq}"
    )
    with name_tracker.set_contexts(contexts_dict=context):
        obs_path_template = str(obs_asset.path)
        cmb_path_template = str(cmb_asset.path)

    dataset = LabelledCMBMapDataset(
        n_sims = split.n_sims,
        freqs = instrument.dets.keys(),
        map_fields=map_fields,
        label_path_template=cmb_path_template, 
        feature_path_template=obs_path_template,
        file_handler=HealpyMap(),
        transforms=transforms
        )
    return dataset
