"""
This module provides functionality to download science assets from original 
remote locations and saves them to a target directory.

Module is intended to be run as a standalone, using the Hydra framework to
manage configuration.
"""
from typing import Union
import os
from pathlib import Path
import requests
import logging
import re
import zipfile
import tarfile

from tqdm import tqdm
import hydra


logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class FileSizeNotFound:
    """Sentinel class to indicate file size information could not be obtained."""
    def __init__(self):
        raise NotImplementedError("This sentinel class should not be instantiated.")
class FileNotFound:
    """Sentinel class to indicate no file found at a URL or on the local disk."""
    def __init__(self):
        raise NotImplementedError("This sentinel class should not be instantiated.")


@hydra.main(version_base=None, config_path="../cfg", config_name="config_get_science_assets")
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


def ensure_dir_exists(asset_dir) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        asset_dir (Path): The location in which assets will be stored.
    """
    if not asset_dir.exists():
        asset_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Created directory: {asset_dir}")
    else:
        logger.debug(f"Directory already exists: {asset_dir}")


def correct_url(url) -> str:
    """
    Ensures the URL starts with a valid scheme and '//' where necessary.

    Args:
        url (str): The URL of the file.
    """
    # Check if URL has the correct http:// or https:// prefix followed by double slashes
    if not re.match(r'https?://', url):
        # Attempt to fix missing or malformed scheme
        url = re.sub(r'^https?:?/', 'http://', url)  # Default to http if unsure
        url = re.sub(r'^https?:?/', 'https://', url) if 'https' in url else url
    # Ensure double slashes follow the scheme
    url = re.sub(r'https?:/', 'http://', url)  # Correct to double slashes for http
    url = re.sub(r'https?:/', 'https://', url) if 'https' in url else url
    return url


def find_remote_size(url) -> Union[int, FileNotFound, FileSizeNotFound]:
    """
    Attempts to retrieve the size of a file from a URL using a HEAD request.

    Args:
        url (str): The URL of the file.

    Returns:
        int or FileNotFound: The size of the file in bytes if it exists,
        or a FileNotFound sentinel if the file does not exist.

    Raises:
        requests.exceptions.RequestException: If an HTTP error occurs during the request.
        FileNotFoundError: If the remote file is not found.
    """
    response = requests.head(url)
    if response.status_code == 404:
        raise FileNotFoundError(f"Remote file not found: {url}")
    response.raise_for_status()  # Will handle other erroneous status codes
    size = response.headers.get('Content-Length')
    if size is not None and size != '0':
        return int(size)
    else:
        # This is expected from some servers, e.g. GitHub
        logging.info(f"Remote file size not found.")
        return FileSizeNotFound


def find_local_size_or_notfound(target_fp) -> Union[int, FileNotFound]:
    """
    Check if the file on disk has the same size as the remote file.

    Args:
        target_fp (str): The file path of the target file on disk.
        remote_file_size (int): The size of the remote file.

    Returns:
        bool: True if the file on disk has the same size as the remote file, False otherwise.
    """
    try:
        local_file_size = os.path.getsize(target_fp)
    except FileNotFoundError:
        # Use a sentinel class to indicate that the file was not found
        #    this is more clear to check than a None return value
        local_file_size = FileNotFound
    return local_file_size


def download_file(url, destination, remote_file_size) -> None:
    """
    Downloads a file from a URL to a specified destination with a progress bar, using logging for error and status messages.

    Args:
    url (str): URL from which to download the file.
    destination (str): Path to save the file to.
    remote_file_size (int): Expected size of the file to be downloaded in bytes.

    Raises:
    requests.exceptions.RequestException: For issues related to the HTTP request.
    Exception: For other issues like file write errors or progress mismatch.
    """
    block_size = 1024  # 1 Kibibyte

    # tqdm will produce the progress bar
    # Initialize tqdm based on whether file size is known
    total = remote_file_size if remote_file_size is not FileSizeNotFound else None
    progress_bar = tqdm(total=total, unit='iB', unit_scale=True, leave=True)

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check that the request was successful
            with open(destination, 'wb') as file:
                # Update the progress bar as data is downloaded
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error occurred during the download: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        progress_bar.close()  # Ensure that progress bar is closed


def extract_file(file_path: Path) -> None:
    """
    Extracts a file to a target directory with a progress bar and error handling.
    Used for the WMAP9 chains.

    Args:
    file_path (Path): Path to the file to extract.
    """
    if file_path.suffixes[-2:] == ['.tar', '.gz']:
        out_dir = file_path.parent / file_path.stem[:-4]  # Removes the last 4 chars, i.e., ".tar"
    else:
        out_dir = file_path.parent / file_path.stem

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extraction logic with progress bar
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())
                with tqdm(total=total_files, desc="Extracting ZIP", unit='files') as pbar:
                    for file in zip_ref.infolist():
                        zip_ref.extract(file, path=out_dir)
                        pbar.update(1)
        elif file_path.suffixes[-1] == '.gz' and file_path.suffixes[-2] == '.tar':
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                total_files = len(tar_ref.getmembers())
                with tqdm(total=total_files, desc="Extracting TAR.GZ", unit='files') as pbar:
                    for member in tar_ref:
                        tar_ref.extract(member, path=out_dir)
                        pbar.update(1)
        logging.info("Extraction completed successfully.")
    except zipfile.BadZipFile:
        logging.error("Failed to extract ZIP file: The file may be corrupted.")
    except tarfile.TarError:
        logging.error("Failed to extract TAR file: The file may be corrupted or it is in an unsupported format.")
    except PermissionError:
        logging.error("Permission denied: Unable to write to the directory.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
