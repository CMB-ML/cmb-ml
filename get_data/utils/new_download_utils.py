from pathlib import Path
import requests
import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

def download_progress(dest_path, source_url_template, file_size=None):
    """Load map data from a file, downloading it if necessary."""
    need_to_dl = check_need_download(dest_path, file_size=file_size)
    fn = dest_path.name
    if need_to_dl:
        response = requests.get(source_url_template.format(fn=fn), stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0 and file_size is not None:
            total_size = file_size * 1000 * 1000  # Convert MB to bytes

        chunk_size = 1024 * 1024  # Download in 1MB chunks

        with open(dest_path, "wb") as file, tqdm(
            desc=f"Downloading {fn}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            position=1
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))


def download(dest_path, source_url_template):
    """Load map data from a file, downloading it if necessary."""
    need_to_dl = check_need_download(dest_path)
    fn = dest_path.name
    if need_to_dl:
        response = requests.get(source_url_template.format(fn=fn))
        with open(dest_path, "wb") as file:
            file.write(response.content)
        logger.info(f"Downloaded {fn}")


def check_need_download(dest_path, file_size=None):
    """Check if a file exists."""
    need_to_dl = False
    if not dest_path.exists():
        logger.info(f"File {dest_path} does not exist; downloading.")
        need_to_dl = True
    elif dest_path.stat().st_size < 1024:  # If the file is less than 1KB, it's a placeholder file
        logger.info(f"File {dest_path} has placeholder file; redownloading.")
        need_to_dl = True
    elif file_size is not None and dest_path.stat().st_size < file_size * 1000 * 1000:
        logger.info(f"File {dest_path} is too small; redownloading.")
        need_to_dl = True
    else:
        logger.debug(f"File {dest_path} exists.")
    return need_to_dl
