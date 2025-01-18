from pathlib import Path
import logging

from get_data.utils.new_download_utils import (
    download,
    download_progress,
)
from get_data.utils.download import extract_file


logger = logging.getLogger(__name__)


def get_wmap_chains_ext(assets_directory, mnu=True, progress=False):
    url_template_maps = "https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/{fn}"
    if mnu:
        fn = "wmap_lcdm_mnu_wmap9_chains_v5.tar.gz"
        file_size = 382  # MB
    else:
        fn = "wmap_lcdm_wmap9_chains_v5.tar.gz"
        file_size = 784  # MB

    dest_path = Path(assets_directory) / fn
    if progress:
        download_progress(dest_path, url_template_maps, file_size=file_size)
    else:
        download(dest_path, url_template_maps)

    logger.info("Files downloaded. Extracting...")

    extract_file(dest_path)

    return dest_path
