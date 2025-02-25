# This module contains helper functions for D_getting_dataset_instances.ipynb
from pathlib import Path
import tempfile

from cmbml.core.asset_handlers import Config
from cmbml.core import Namer
from cmbml.get_data.utils.get_from_shared_link import download_shared_link_info



def get_simulation(cfg, dataset_json_fn, split, sim_num):
    json_dir = Path(cfg.local_system.assets_dir) / "CMB-ML"
    json_path = json_dir / dataset_json_fn

    config_reader = Config()
    all_shared_links = config_reader.read(json_path)
    key = f"{split}_sim{sim_num:04d}"
    shared_link = all_shared_links[key]
    
    # TODO: Using Namer twice is clunky, come up with better solution.
    # We set up a Namer to get the root directory of the dataset
    #   because the tar includes directory structure
    namer = Namer(cfg)
    path_template = "{root}/{dataset}"
    context_params = dict(
        dataset=cfg.dataset_name
    )
    with namer.set_contexts(context_params):
        dest = namer.path(path_template)

    with tempfile.TemporaryDirectory() as temp_tar_dir:
        download_shared_link_info(shared_link, temp_tar_dir, dest)

    # This second Namer will give the path to the particular simulation
    namer = Namer(cfg)
    path_template = cfg.file_system.default_dataset_template_str
    context_params = dict(
        dataset=cfg.dataset_name,
        stage="Simulation",
        split=split,
        sim_num=sim_num
    )
    with namer.set_contexts(context_params):
        dest = namer.path(path_template)

    return dest
