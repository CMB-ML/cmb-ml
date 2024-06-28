"""
This module provides functionality to download science assets the box repository.

Module is intended to be run as a standalone, using the Hydra framework to
manage configuration.
"""
import logging
import os
from pathlib import Path
from dataclasses import asdict
import json

import hydra

from .utils.box_links import get_box_links_from_json, FromBoxDownloader, LinkInfo


logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../cfg", config_name="config_demo_dataset")
def main(cfg):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    force_download = cfg.force_download

    destination = Path(cfg.local_system.assets_dir)

    json_file = "./box_link_jsons/box_info_Assets.json"
    json_file = Path(json_file)

    box_links = get_box_links_from_json(json_file, compose_path=True)

    downloader = FromBoxDownloader(destination_root=destination,
                                   do_untar=False)
    dl_box_links_rec(downloader, box_links)


def dl_box_links_rec(downloader: FromBoxDownloader, box_links):
    for link_info in box_links.values():
        if isinstance(link_info, LinkInfo):
            downloader.download(link_info)
        else:
            dl_box_links_rec(downloader, link_info)


if __name__ == "__main__":
    main()
