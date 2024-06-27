"""
This module provides functionality to download the I_128_1450 dataset from the box repository.

This module is intended to be run as a standalone, using the Hydra framework to
manage configurations.
"""
import logging
import os
from pathlib import Path

import hydra
from tqdm import tqdm

from cmbml.utils.box_links import get_box_links_from_json, FromBoxDownloader


logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../cfg", config_name="config_demo_dataset")
def main(cfg):
    # Need to change directory to the location of this script to ensure that the
    # relative path of the json file is correct.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cfg.nside = 128
    cfg.map_fields = "I"

    dataset_name = cfg.dataset_name
    json_file = f"./box_link_jsons/box_info_{dataset_name}.json"
    json_file = Path(json_file)

    dest = Path(cfg.local_system.datasets_root) / dataset_name
    downloader = FromBoxDownloader(destination_root=dest,
                                   remove_tar=True)

    box_links = get_box_links_from_json(json_file, compose_path=False)

    for link_info in tqdm(box_links.values()):
        downloaded = downloader.download(link_info)
        if not downloaded:
            logger.error(f"Failed to download {link_info.file_name} from Box.")
            break


if __name__ == "__main__":
    main()
