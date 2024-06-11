from typing import Dict
from pathlib import Path
import json
import requests


def load_shared_links(json_path: Path) -> Dict:
    with open(json_path, 'r') as f:
        shared_links = json.load(f)
    return shared_links


def is_leaf(some_dict):
    res = True
    for v in some_dict.values():
        if isinstance(v, dict):
            res = False
            break
    return res


def add_true_urls(shared_links: Dict) -> Dict:
    for node in shared_links.values():
        if is_leaf(node):
            node['url'] = unbox_url(node)
        else:
            add_true_urls(node)
    return 


def unbox_url(shared_link) -> str:
    box_id = shared_link['box_id']
    obf_url = shared_link['shared_link']
    shared_name = obf_url.split('/')[-1]
    url = f"https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name={shared_name}&file_id=f_{box_id}"
    return url


def download_files(true_links: Dict, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for k, v in true_links.items():
        if is_leaf(v):
            url = v['url']
            target_path = target_dir / k
            print(f"Downloading {k} from {url}")
            download_file(url, target_path)
        else:
            download_files(v, target_dir / k)


def download_file(url: str, destination: Path) -> None:
    """
    Downloads a file from the given URL and saves it to the specified destination.

    Args:
        url (str): The URL of the file to download.
        destination (Path): The path where the downloaded file should be saved.

    Raises:
        Exception: If the download fails or the response status code is not 200.

    """
    with requests.get(url, stream=True, allow_redirects=True) as response:
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            raise Exception(f"Failed to download file from {url} - Status code: {response.status_code}")
