from pathlib import Path

from .box_download_utils import make_url_from_shared_link, download_file, extract_tar_file
from .get_sha import calculate_sha1


def download_shared_link_info(shared_link_info: dict, 
                              temp_tar_dir: Path,
                              dataset_dir: Path) -> None:
    temp_tar_dir = Path(temp_tar_dir)
    dataset_dir = Path(dataset_dir)
    temp_tar_dir.mkdir(parents=True, exist_ok=True)

    shared_link = shared_link_info
    tar_file_name = shared_link["file_name"]
    tar_file_size = shared_link.get("file_size", None)
    tar_path = temp_tar_dir / tar_file_name

    url = make_url_from_shared_link(shared_link)
    t_pos = 1  # Operating within the tqdm progress bar from execute()
    download_file(url, tar_path, filesize=tar_file_size, tqdm_position=t_pos)

    new_sha1 = calculate_sha1(tar_path)
    old_sha1 = shared_link["archive_sha1"]

    if new_sha1 != old_sha1:
        raise ValueError(f"SHA1 checksums do not match for file {tar_file_name}.")
    
    extract_tar_file(tar_path, dataset_dir)
    tar_path.unlink()
