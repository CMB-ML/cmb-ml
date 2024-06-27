from pathlib import Path
import json
import requests
import tarfile
import hashlib
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


def get_box_links_from_json(json_path: Path, compose_path) -> dict:
    """
    Create a dictionary with required information for all files to download.

    Args:
        json_path (Path): Path to the json file containing the box links.
        compose_path (bool): If True, the file_name will be composed from the directory structure in the json file.
                                (For use with Science Assets, where the directory structure is the same as on Box.)
                             If False, the file_name will be taken from the json file.
                                (For use with Datasets, which are in tar files that preserve directory structure.)

    Returns:
        box_links: A dictionary with the BoxInfo objects as values, with
                   nested dictionary structure matching JSON structure.
    """
    with open(json_path, 'r') as f:
        box_links_dict = json.load(f)
    box_links = _process_box_links_dict(box_links_dict, compose_path)
    return box_links


def _process_box_links_dict(d, compose_path: bool, path=[]):
    """
    Recursively process the nested dictionary from the JSON to create a dictionary.

    Args:
        d (dict): The nested dictionary from the JSON.
        compose_path (bool): If True, the file_name will be composed from the directory structure in the json file.
                                (For use with Science Assets, where the directory structure is the same as on Box.)
                             If False, the file_name will be taken from the json file.
                                (For use with Datasets, which are in tar files that preserve directory structure.)
        path (list): The path to the current node in the nested dictionary.
    
    Returns:
        box_links: A dictionary with the BoxInfo objects as values, with
                   nested dictionary structure matching JSON structure
    """
    if _is_leaf(d):
        if compose_path:
            # For science assets (directory structure is same as on Box)
            file_name = str(Path(*path))
        else:
            # For datasets (in tar files, which preserve directory structure)
            file_name = d['file_name']
        d = {k: v for k, v in d.items() if k != 'file_name'}
        return LinkInfo(file_name=file_name, **d)
    else:
        return {k: _process_box_links_dict(v, compose_path, path + [k]) for k, v in d.items()}


def _is_leaf(some_dict):
    """
    Check if a dictionary is a leaf node in the nested dictionary from the JSON, indicated by the absence of nested
    dictionaries.

    Returns:
        bool: True if the dictionary is a leaf node, False otherwise
    """
    res = True
    for v in some_dict.values():
        if isinstance(v, dict):
            res = False
            break
    return res


@dataclass
class LinkInfo:
    """
    Dataclass to store information about a file to download from Box.
    """
    file_name: str  # Name of the file
    box_id: str     # Box file ID
    token: str      # Box shared link token
    md5: str        # MD5 checksum of the file

    def generate_url(self):
        base_url = "https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{box_id}"
        url = base_url.format(token=self.token, box_id=self.box_id)
        return url


class FromBoxDownloader:
    """
    Class to download files from Box using the LinkInfo.
    """
    def __init__(self, 
                 destination_root: Path, 
                 do_untar: bool=True, 
                 remove_tar: bool=True):
        self.destination_root = destination_root
        self.do_untar = do_untar
        self.remove_tar = remove_tar

    def download(self, link_info: LinkInfo, rel_path: str=None):
        """
        Download a file from Box using the LinkInfo object.

        Args:
            link_info (LinkInfo): Information about the file to download.
            rel_path (str): Relative path to the file in the destination directory.

        Returns:
            bool: True if the file was downloaded successfully, False otherwise.
        """
        if self.do_untar and is_tar_gz_file(link_info.file_name):
            dest_fn = Path(link_info.file_name).name
        else:
            dest_fn = link_info.file_name

        dest_path = self.destination_root / dest_fn

        if not dest_path.parent.exists():
            dest_path.parent.mkdir(exist_ok=True, parents=True)

        url = link_info.generate_url()

        try:
            download_file(url, dest_path)
        except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
            logger.error(f"Error downloading file: {e}", exc_info=True)
            return False
        except IOError as e:
            logger.error(f"Error writing file: {e}", exc_info=True)
            return False

        try:
            self.compare_checksums(dest_path, link_info.md5)
        except ValueError as e:
            logger.error(f"Error comparing checksums: {e}", exc_info=True)
            return False

        if self.do_untar and is_tar_gz_file(link_info.file_name):
            self.untar_file(dest_path)
            if self.remove_tar:
                self.remove_downloaded_tar(dest_path)
        return True

    def compare_checksums(self, file_path: Path, old_checksum:str):
        new_checksum = md5_checksum(file_path)
        if new_checksum != old_checksum:
            raise ValueError(f"Checksum mismatch for {file_path.name}")

    def untar_file(self, tar_path):
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.destination_root)
        except Exception as e:
            print(f"Error unpacking tar file: {e}")

    def remove_downloaded_tar(self, tar_path: Path):
        tar_path.unlink()


def download_file(url, dest_path):
    """
    Downloads a file from some URL and saves it to a specified destination.

    The file is downloaded in chunks to manage memory usage efficiently.

    Args:
        url (str): Remote location of the asset.
        dest_path (str or Path): The file system path where the downloaded file should be saved.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
        requests.exceptions.RequestException: For network-related errors.
        IOError: If there are issues writing the file to disk.
    """
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()  # Note the exception(s) raised in the docstring
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return True


def md5_checksum(file_path):
    # Create the hash object
    hash_md5 = hashlib.md5()
    # Open the file in binary mode
    with open(file_path, "rb") as f:
        # Read in chunks to manage memory
        for chunk in iter(lambda: f.read(4096), b""):
            # Update the hash with the chunk
            hash_md5.update(chunk)
    # Return the hexadecimal checksum
    return hash_md5.hexdigest()


def is_tar_gz_file(file_path):
    path = Path(file_path)
    # Check if the last two suffixes are '.tar' and '.gz'
    return path.suffixes[-2:] == ['.tar', '.gz']
