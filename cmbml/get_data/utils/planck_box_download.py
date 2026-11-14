"""Download public Box assets using their Box IDs and SHA-1 checksums.

Uses the public folder's sharing token; no Box account is necessary.
Requires the `rclone` executable on PATH (no configured remote necessary).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess


logger = logging.getLogger(__name__)
_BOX_ID = re.compile(r"[0-9]+\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9]+\Z")


def _sha1(path: Path) -> str:
    checksum = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Planck manifest format: {path}")
    if not _TOKEN.fullmatch(manifest.get("folder_shared_token", "")):
        raise ValueError("Invalid public Box folder token")
    names = set()
    for entry in manifest["files"]:
        name = entry["name"]
        if (not name or name in (".", "..") or name != Path(name).name or "\\" in name
                or name in names):
            raise ValueError(f"Invalid or repeated filename: {name!r}")
        names.add(name)
        if not _BOX_ID.fullmatch(entry["box_id"]):
            raise ValueError(f"Invalid Box ID for {name}")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise ValueError(f"Invalid file size for {name}")
        if not _SHA1.fullmatch(entry["sha1"]):
            raise ValueError(f"Invalid SHA-1 for {name}")
    return manifest


def download_planck_assets(manifest_path: Path, destination: Path) -> None:
    """Download every listed asset; only accept matching size and SHA-1.

    Files are written to .partial paths and moved into place after verification.
    Invalid existing files (including small HTTP error pages) are replaced.
    """
    manifest = _read_manifest(manifest_path)
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is required to download Planck assets from Box")
    destination.mkdir(parents=True, exist_ok=True)
    token = manifest["folder_shared_token"]
    url_pattern = manifest["download_url_pattern"]
    for entry in manifest["files"]:
        name = entry["name"]
        target = destination / name
        if (target.is_file() and target.stat().st_size == entry["size_bytes"]
                and _sha1(target) == entry["sha1"]):
            logger.info("Already verified: %s", name)
            continue

        logger.info("Downloading Box asset: %s", name)
        partial = destination / (name + ".partial")
        partial.unlink(missing_ok=True)  # rclone copyurl does not resume partial bytes.
        url = url_pattern.format(folder_shared_token=token, box_id=entry["box_id"])
        try:
            subprocess.run(
                ["rclone", "copyurl", url, str(partial), "--retries", "3",
                 "--low-level-retries", "10", "--progress"],
                check=True,
            )
            if not partial.is_file() or partial.stat().st_size != entry["size_bytes"]:
                actual = partial.stat().st_size if partial.exists() else "missing"
                raise RuntimeError(f"Incorrect size for {name}: {actual} (expected {entry['size_bytes']})")
            actual_sha1 = _sha1(partial)
            if actual_sha1 != entry["sha1"]:
                raise RuntimeError(f"SHA-1 mismatch for {name}: {actual_sha1}")
            os.replace(partial, target)
            logger.info("Verified: %s", name)
        finally:
            partial.unlink(missing_ok=True)
