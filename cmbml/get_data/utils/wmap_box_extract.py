"""Safely extract verified WMAP9 tarballs to their expected chain directories."""
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

logger = logging.getLogger(__name__)


def extract_wmap_archive(archive: Path) -> Path:
    """Extract an archive next to itself, skipping an already complete extraction.

    Handles archives whose contents are either flat or inside one parent folder.
    A failed extraction is retried the next time this function is called.
    """
    archive = Path(archive)
    if not archive.name.endswith('.tar.gz'):
        raise ValueError(f"Not a .tar.gz file: {archive}")
    target = archive.parent / archive.name.removesuffix('.tar.gz')

    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError(f"Empty archive: {archive}")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsupported or unsafe tar member: {member.name!r}")

        member_paths = [PurePosixPath(m.name) for m in members]
        tops = {p.parts[0] for p in member_paths if p.parts}
        # If the tar contains exactly one enclosing directory, strip it.
        enclosed = (len(tops) == 1
                    and any(len(p.parts) > 1 for p in member_paths)
                    and all(len(p.parts) > 1 or m.isdir()
                            for m, p in zip(members, member_paths)))
        prefix = next(iter(tops)) if enclosed else None

        expected = []
        for member, path in zip(members, member_paths):
            if not member.isfile():
                continue
            relative = PurePosixPath(*path.parts[1:]) if enclosed else path
            if not relative.parts:
                raise ValueError(f"Unexpected tar member: {member.name!r}")
            expected.append((Path(*relative.parts), member.size))
        if not expected:
            raise ValueError(f"No files in archive: {archive}")

        if target.is_dir() and all(
            (target / name).is_file() and (target / name).stat().st_size == size
            for name, size in expected
        ):
            logger.info("WMAP chains already extracted: %s", target)
            return target

        logger.info("Extracting WMAP chains: %s", archive.name)
        with tempfile.TemporaryDirectory(prefix='.wmap_extract_', dir=archive.parent) as tmp:
            staging = Path(tmp)
            tar.extractall(path=staging, filter='data')
            source = staging / prefix if enclosed else staging
            if not source.is_dir():
                raise RuntimeError(f"Missing extracted directory in {archive}")
            # Copying only after a complete extraction avoids exposing a truncated tar.
            shutil.copytree(source, target, dirs_exist_ok=True)

        if not all(
            (target / name).is_file() and (target / name).stat().st_size == size
            for name, size in expected
        ):
            raise RuntimeError(f"Extracted files incomplete: {archive}")
    logger.info("Extracted and checked: %s", target)
    return target
