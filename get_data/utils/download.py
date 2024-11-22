"""
This module contains downloading utilities for conventional links and for Box links, both in custom formats.
"""
import hashlib
import logging
import os
from pathlib import Path
import re
import requests
import tarfile
from typing import Union
import zipfile

from tqdm import tqdm


logger = logging.getLogger(__name__)


class FileSizeNotFound:
    """Sentinel class to indicate file size information could not be obtained."""
    def __init__(self):
        raise NotImplementedError("This sentinel class should not be instantiated.")
class FileNotFound:
    """Sentinel class to indicate no file found at a URL or on the local disk."""
    def __init__(self):
        raise NotImplementedError("This sentinel class should not be instantiated.")


def download_file(url, destination, remote_file_size, block_size=16384) -> None:
    """
    Downloads a file from a URL to a specified destination with a progress bar, using logging for error and status messages.

    Args:
        url (str): URL from which to download the file.
        destination (str): Path to save the file to.
        remote_file_size (int): Expected size of the file to be downloaded in bytes.
        block_size (int): Size of the blocks to download the file in.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
        requests.exceptions.RequestException: For network-related errors.
        IOError: If there are issues writing the file to disk.
    """
    # tqdm will produce the progress bar
    # Initialize tqdm based on whether file size is known
    total = remote_file_size if remote_file_size is not FileSizeNotFound else None
    progress_bar = tqdm(total=total, unit='iB', unit_scale=True, leave=False)

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as file:
                # Update the progress bar as data is downloaded
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
    finally:
        progress_bar.close()  # Ensure that progress bar is closed


def md5_checksum(file_path):
    """
    Computes the MD5 checksum of a file on the local system.
    """
    hash_md5 = hashlib.md5()
    # Open the file in binary mode
    with open(file_path, "rb") as f:
        # Read in chunks to manage memory
        for chunk in iter(lambda: f.read(4096), b""):
            # Update the hash with the chunk
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_tar_gz_file(file_path: Union[Path, str]):
    """
    Determines if a file path is end with ".tar.gz". Does not check existence or anything about the actual file contents.
    """
    # Ensure it's a path
    path = Path(file_path)
    # Check if the last two suffixes are '.tar' and '.gz'
    return path.suffixes[-2:] == ['.tar', '.gz']


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
        # This is expected from some servers, e.g. GitHub, Box
        logging.debug(f"Remote file size not found.")
        return FileSizeNotFound


def extract_file(file_path: Path) -> None:
    """
    Extracts a file to a target directory with a progress bar and error handling.
    Used for the WMAP9 chains.

    Args:
    file_path (Path): Path to the file to extract.
    """
    if is_tar_gz_file(file_path):
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
                with tqdm(total=total_files, desc="Extracting", unit='files') as pbar:
                    for file in zip_ref.infolist():
                        zip_ref.extract(file, path=out_dir)
                        pbar.update(1)
        elif file_path.suffixes[-1] == '.gz' and file_path.suffixes[-2] == '.tar':
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                total_files = len(tar_ref.getmembers())
                with tqdm(total=total_files, desc="Extracting", unit='files') as pbar:
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
