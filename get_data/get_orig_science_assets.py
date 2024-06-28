"""
This module provides functionality to download science assets from original 
remote locations and saves them to a target directory.

Module is intended to be run as a standalone, using the Hydra framework to
manage configuration.
"""
from pathlib import Path
import logging

import hydra

from get_data.utils.download import (
    download_file, 
    find_local_size_or_notfound, 
    extract_file, 
    find_remote_size, 
    ensure_dir_exists,
    FileNotFound
)


logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../cfg", config_name="config_get_orig_science_assets")
def main(cfg):
    force_download = cfg.force_download
    assets_to_get = cfg.assets_to_get

    for assets in assets_to_get:
        mission_dir = assets.mission_dir
        base_url = assets.base_url
        match_size = assets.get("match_size", True)
        extract = assets.get("extract", False)
        logger.debug(f"Getting assets for {mission_dir}")
        target_dir = Path(cfg.local_system.assets_dir) / mission_dir
        ensure_dir_exists(target_dir)
        for file in assets.files:
            get_science_asset(file, base_url, target_dir, match_size, extract, force_download)
    logger.info("Assets acquired.")


def get_science_asset(file: str,
                      base_url: str,
                      target_dir: Path,
                      match_size: bool,
                      extract: bool,
                      force_download: bool):
    """
    Downloads a science asset from a remote location and saves it to the target directory.

    Args:
        file (str): The name of the file to download.
        base_url (str): The base URL of the remote location.
        target_dir (Path): The directory where the file will be saved.
        match_size (bool): If True, checks if the local file size matches the remote file size before downloading.
        extract (bool): If True, extracts the file if it is a compressed archive.
        force_download (bool): If True, forces the download even if the file already exists locally.

    Returns:
        None
    """
    # Get locations 
    # pathlib doesn't work for URLs, so string assemble it:
    url = base_url + '/' + file
    # Use pathlib for local file paths:
    target_fp = target_dir / file

    # Determine existence and size of remote file
    try:
        remote_size = find_remote_size(url)
    except:
        logging.warning(f"Skipping. Remote file not found: {url}")
        return

    # Determine if download is needed
    local_size = find_local_size_or_notfound(target_fp)
    need_download = False
    if force_download:
        need_download = True
    elif local_size is FileNotFound:
        need_download = True
    elif match_size and (local_size != remote_size):
        need_download = True

    if need_download:
        logger.info(f"Downloading {url}")
        download_file(url, target_fp, remote_size)
    else:
        logger.info(f"File on disk. Skipping {url}")

    # If the file is a compressed archive, extract it
    if extract:
        logger.info(f"Extracting {target_fp}")
        extract_file(target_fp)


if __name__ == "__main__":
    main()
