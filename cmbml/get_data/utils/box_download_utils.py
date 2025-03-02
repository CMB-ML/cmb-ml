"""
This module provides utility functions for downloading and extracting files from Box shared links.

Functions:
- load_shared_links(json_file_path: str) -> dict: Load shared links from a JSON file.
- download_file(url: str, destination: Path) -> None: Download a file from a URL to a specified destination, handling redirects.
- extract_tar_file(tar_path, extract_to): Extract a tar.gz file to a specified directory.
- make_url(shared_link): Construct the download URL for a Box shared link.
"""

from pathlib import Path
import tarfile
import json
import requests
import logging

from tqdm import tqdm


logger = logging.getLogger(__name__)


def load_shared_links(json_file_path: str) -> dict:
    """
    Load shared links from a JSON file.

    Args:
        json_file_path (str): The path to the JSON file containing the shared links.

    Returns:
        dict: A dictionary containing the shared links.

    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


def download_file(url: str, destination: Path, filesize: int = None, tqdm_position: int = 1) -> None:
    """
    Downloads a file from the given URL and saves it to the specified destination with a progress bar.

    Args:
        url (str): The URL of the file to download.
        destination (Path): The path where the downloaded file should be saved.
        filesize (int, optional): The total size of the file in bytes. If not provided, it will be determined from the response.
        position (int, optional): The row position of the progress bar in a nested tqdm setup.

    Raises:
        Exception: If the download fails or the response status code is not 200.
    """
    with requests.get(url, stream=True, allow_redirects=True) as response:
        if response.status_code == 200:
            # Determine the total file size for progress bar
            total_size = int(response.headers.get('content-length', 0)) if filesize is None else filesize
            chunk_size = 8192  # 8 KB chunks
            with open(destination, 'wb') as file, tqdm(
                desc=f"Downloading {destination.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                position=tqdm_position,
                leave=False,  # Inner bar disappears after completion
            ) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    progress.update(len(chunk))
        else:
            raise Exception(f"Failed to download file from {url} - Status code: {response.status_code}")


def extract_tar_file(tar_path: Path, extract_to: Path) -> None:
    """
    Extracts the contents of a tar file to a specified directory.

    Parameters:
    tar_path (str): The path to the tar file.
    extract_to (str): The directory where the contents of the tar file will be extracted to.

    Returns:
    None
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=f"Extracting {tar_path.name}", unit="file", leave=False) as progress:
            for member in members:
                tar.extract(member, path=extract_to)
                progress.update(1)


def make_url_from_shared_link(box_info_dict: dict) -> str:
    """
    Box provides links to a website, where the file can be downloaded. This function constructs the download URL.

    Args:
        shared_link (dict): A dictionary containing the Box ID and shared link.

    Returns:
        str: The download URL for the shared file.

    """
    box_id = box_info_dict['box_id']
    token = box_info_dict['token']
    url = make_url(box_id, token)
    return url


def make_url(box_id, token):
    url = f"https://removed.app.box.com/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{box_id}"
    return url
