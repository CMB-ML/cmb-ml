# This module contains helper functions for D_getting_dataset_instances.ipynb
from pathlib import Path
import tempfile

from cmbml.core.asset_handlers import Config
from cmbml.core import Namer
from get_data.utils.get_from_shared_link import download_shared_link_info



def get_simulation(cfg, dataset_json_fn, split, sim_num):
    json_dir = Path(cfg.local_system.assets_dir) / "CMB-ML"
    json_path = json_dir / dataset_json_fn

    config_reader = Config()
    all_shared_links = config_reader.read(json_path)
    key = f"{split}_sim{sim_num:04d}"
    shared_link = all_shared_links[key]

    namer = Namer(cfg)
    path_template = "{root}/{dataset}"
    context_params = dict(
        dataset=cfg.dataset_name
    )
    with namer.set_contexts(context_params):
        dest = namer.path(path_template)

    with tempfile.TemporaryDirectory() as temp_tar_dir:
        download_shared_link_info(shared_link, temp_tar_dir, dest)

    return dest
