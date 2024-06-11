"""
This module provides functionality to download science assets the box repository.

Module is intended to be run as a standalone, using the Hydra framework to
manage configuration.
"""
from pathlib import Path
import logging

import hydra

from .use_shared_links import (
    load_shared_links, 
    add_true_urls, 
    download_files
)


logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="cfg", config_name="config_get_box_science_assets")
def main(cfg):
    destination = Path(cfg.local_system.assets_dir)

    shared_links = load_shared_links(cfg.shared_links)
    add_true_urls(shared_links)
    download_files(shared_links, destination)


if __name__ == "__main__":
    main()
