#!/usr/bin/env python3
"""
Extract a shard set created by make_tar_shards.py.

The destination argument is the exact directory that will become the restored
copy of the dataset. Archive paths begin with the original source directory
name; that leading component is removed during extraction, so you may choose a
new directory name for the restored copy.

Example:

    python3 extract_tar_shards.py \
        /data/jim/CMB_Data/Datasets/I_512_1850_broad_dist_shards \
        /data/jim/CMB_Data/Datasets/I_512_1850_broad_dist_restored

Features:
  * sequential extraction with live percent, throughput, elapsed time, and ETA
  * SHA-256 verification while each tar is being extracted (one pass)
  * validation of every tar member against manifest.jsonl
  * atomic file replacement via a temporary directory outside the restored tree
  * an external state file for automatic restart/resume after interruption
  * final restoration of directory modification times from the manifest

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence


SCRIPT_VERSION = 1
BUFFER_SIZE = 8 * 1024 * 1024
SHA256_RE = re.compile(r"^([0-9a-fA-F]{64})\s+[* ]?(.+)$")


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpectedMember:
    archive_path: str
    relative_path: str
    kind: str
    size: int
    shard: int
    mtime_ns: int | None


class HashingReader:
    """Read-through wrapper that computes SHA-256 for the complete tar file."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        if data:
            self.hasher.update(data)
            self.bytes_read += len(data)
        return data

    def close(self) -> None:
        self.raw.close()


class ProgressReporter:
    """Low-overhead terminal progress with current-shard and overall ETA."""

    def __init__(
        self,
        *,
        shard_number: int,
        shard_count: int,
        shard_total: int,
        overall_base: int,
        overall_resume_base: int,
        overall_total: int,
        overall_started: float,
        interval: float,
    ) -> None:
        self.shard_number = shard_number
        self.shard_count = shard_count
        self.shard_total = max(int(shard_total), 0)
        self.overall_base = max(int(overall_base), 0)
        self.overall_resume_base = max(int(overall_resume_base), 0)
        self.overall_total = max(int(overall_total), 0)
        self.overall_started = overall_started
        self.interval = max(float(interval), 0.1)
        self.completed = 0
        self.started = time.monotonic()
        self.last_printed = self.started
        self.last_width = 0
        self.tty = sys.stderr.isatty()
        self.current_path = ""

    def set_path(self, relative_path: str) -> None:
        self.current_path = relative_path

    def _line(self, now: float) -> str:
        shard_done = min(self.completed, self.shard_total)
        shard_fraction = (
            shard_done / self.shard_total if self.shard_total else 1.0
        )
        shard_elapsed = max(now - self.started, 1e-9)
        shard_rate = shard_done / shard_elapsed
        shard_remaining = max(self.shard_total - shard_done, 0)
        shard_eta = shard_remaining / shard_rate if shard_rate > 0 else None

        overall_done = min(self.overall_base + shard_done, self.overall_total)
        overall_fraction = (
            overall_done / self.overall_total if self.overall_total else 1.0
        )
        overall_elapsed = max(now - self.overall_started, 1e-9)
        session_done = max(overall_done - self.overall_resume_base, 0)
        overall_rate = session_done / overall_elapsed
        overall_remaining = max(self.overall_total - overall_done, 0)
        overall_eta = (
            overall_remaining / overall_rate if overall_rate > 0 else None
        )

        path_text = self.current_path
        if len(path_text) > 54:
            path_text = "..." + path_text[-51:]

        line = (
            f"Shard {self.shard_number}/{self.shard_count}: "
            f"{shard_fraction * 100:6.2f}% "
            f"({human_bytes(shard_done)}/{human_bytes(self.shard_total)})  "
            f"{human_bytes(shard_rate)}/s  "
            f"ETA {human_duration(shard_eta)}  |  "
            f"overall {overall_fraction * 100:6.2f}% "
            f"ETA {human_duration(overall_eta)}"
        )
        if path_text:
            line += f"  {path_text}"
        return line

    def update(self, amount: int = 0, *, force: bool = False) -> None:
        self.completed += int(amount)
        now = time.monotonic()
        if not force and now - self.last_printed < self.interval:
            return

        line = self._line(now)
        if self.tty:
            padded = line.ljust(self.last_width)
            sys.stderr.write("\r" + padded)
            sys.stderr.flush()
            self.last_width = max(self.last_width, len(line))
        else:
            print(line, file=sys.stderr, flush=True)
        self.last_printed = now

    def finish(self) -> None:
        self.completed = max(self.completed, self.shard_total)
        self.update(force=True)
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def human_bytes(value: int | float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{value} B"


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--"
    seconds = int(round(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExtractionError(f"Required file is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionError(f"Expected a JSON object in {path}")
    return value


def load_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ExtractionError(f"Required file is missing: {path}") from exc
    except OSError as exc:
        raise ExtractionError(f"Could not read {path}: {exc}") from exc

    result: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = SHA256_RE.fullmatch(line)
        if not match:
            raise ExtractionError(
                f"Invalid SHA256SUMS line {line_number}: {line!r}"
            )
        digest = match.group(1).lower()
        filename = match.group(2)
        if filename in result:
            raise ExtractionError(f"Duplicate checksum entry for {filename}")
        result[filename] = digest
    return result


def load_manifest(
    path: Path,
) -> tuple[list[ExpectedMember], str, dict[int, list[ExpectedMember]]]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ExtractionError(f"Required file is missing: {path}") from exc
    except OSError as exc:
        raise ExtractionError(f"Could not read {path}: {exc}") from exc

    digest = hashlib.sha256(raw).hexdigest()
    members: list[ExpectedMember] = []
    by_shard: dict[int, list[ExpectedMember]] = {}

    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
            kind = str(record["type"])
            if kind not in {"file", "directory"}:
                raise ValueError(f"unsupported type {kind!r}")
            mtime_key = "file_mtime_ns" if kind == "file" else "directory_mtime_ns"
            member = ExpectedMember(
                archive_path=str(record["archive_path"]),
                relative_path=str(record["relative_path"]),
                kind=kind,
                size=int(record["size"]),
                shard=int(record["shard"]),
                mtime_ns=(
                    None
                    if record.get(mtime_key) is None
                    else int(record[mtime_key])
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExtractionError(
                f"Invalid manifest record at line {line_number}: {exc}"
            ) from exc

        if member.size < 0 or member.shard <= 0:
            raise ExtractionError(
                f"Invalid size or shard number at manifest line {line_number}"
            )
        members.append(member)
        by_shard.setdefault(member.shard, []).append(member)

    return members, digest, by_shard


def planned_shards(plan: dict) -> list[dict]:
    records = plan.get("shards")
    if not isinstance(records, list) or not records:
        raise ExtractionError("shard_plan.json has no shard records")

    normalized: list[dict] = []
    seen_indices: set[int] = set()
    seen_names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ExtractionError("Invalid shard record in shard_plan.json")
        try:
            index = int(record["index"])
            filename = str(record["filename"])
            payload_bytes = int(record["payload_bytes"])
            member_count = int(record["member_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtractionError(f"Invalid shard record: {record!r}") from exc

        if index in seen_indices:
            raise ExtractionError(f"Duplicate shard index: {index}")
        if filename in seen_names:
            raise ExtractionError(f"Duplicate shard filename: {filename}")
        if index <= 0 or payload_bytes < 0 or member_count < 0:
            raise ExtractionError(f"Invalid shard metadata: {record!r}")
        if not filename.endswith(".tar"):
            raise ExtractionError(f"Planned shard is not a .tar file: {filename}")

        seen_indices.add(index)
        seen_names.add(filename)
        normalized.append(record)

    normalized.sort(key=lambda item: int(item["index"]))
    expected_indices = list(range(1, len(normalized) + 1))
    actual_indices = [int(item["index"]) for item in normalized]
    if actual_indices != expected_indices:
        raise ExtractionError(
            f"Shard indices are not contiguous: {actual_indices[:10]}..."
        )
    return normalized


def validate_relative_path(text: str) -> Path:
    if text == "":
        return Path()
    posix = PurePosixPath(text)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ExtractionError(f"Unsafe relative path in manifest: {text!r}")
    return Path(*posix.parts)


def destination_is_nonempty(destination: Path) -> bool:
    if not destination.exists():
        return False
    if not destination.is_dir():
        raise ExtractionError(f"Destination exists but is not a directory: {destination}")
    try:
        next(destination.iterdir())
    except StopIteration:
        return False
    return True


def default_state_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.extract_tar_shards_state.json"


def default_temp_root(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.extract_tar_shards_tmp"


def initialize_or_load_state(
    *,
    state_path: Path,
    destination: Path,
    plan_id: str,
    shard_count: int,
    force: bool,
) -> dict:
    if state_path.exists():
        state = load_json(state_path)
        expected = {
            "plan_id": plan_id,
            "destination": str(destination),
            "total_shards": shard_count,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ExtractionError(
                    f"Existing extraction state does not match this run ({key}).\n"
                    f"State file: {state_path}"
                )
        completed = state.get("completed_shards")
        if not isinstance(completed, list) or not all(
            isinstance(item, int) and item > 0 for item in completed
        ):
            raise ExtractionError(f"Invalid completed_shards in {state_path}")
        return state

    if destination_is_nonempty(destination) and not force:
        raise ExtractionError(
            "The destination directory is nonempty and no matching resume state "
            "exists. Use a fresh/empty destination, or pass --force only if you "
            "intend to overwrite matching files."
        )

    state = {
        "format": "extract-tar-shards-state",
        "script_version": SCRIPT_VERSION,
        "plan_id": plan_id,
        "destination": str(destination),
        "total_shards": shard_count,
        "completed_shards": [],
        "complete": False,
        "started_unix": time.time(),
        "updated_unix": time.time(),
    }
    atomic_write_json(state_path, state)
    return state


def tar_member_kind(info: tarfile.TarInfo) -> str:
    if info.isdir():
        return "directory"
    if info.isfile():
        return "file"
    return "unsupported"


def safe_target(destination: Path, relative_path: str) -> Path:
    rel = validate_relative_path(relative_path)
    target = destination / rel
    # lexical checks above are sufficient because symlinks are never created or
    # accepted. resolve() is intentionally avoided so nonexistent targets work.
    return target


def write_file_atomically(
    *,
    source: BinaryIO,
    target: Path,
    temp_root: Path,
    relative_path: str,
    expected_size: int,
    mode: int,
    mtime_ns: int | None,
    progress: ProgressReporter,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = temp_root / validate_relative_path(str(Path(relative_path).parent))
    temp_parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.extract-",
        dir=temp_parent,
    )
    temp_path = Path(temp_name)
    written = 0

    try:
        with os.fdopen(fd, "wb") as output:
            while True:
                block = source.read(BUFFER_SIZE)
                if not block:
                    break
                output.write(block)
                written += len(block)
                progress.update(len(block))
            output.flush()
            os.fsync(output.fileno())

        if written != expected_size:
            raise ExtractionError(
                f"Short read for {relative_path}: expected {expected_size:,} bytes, "
                f"read {written:,}"
            )

        os.chmod(temp_path, mode & 0o7777)
        if mtime_ns is not None:
            os.utime(temp_path, ns=(mtime_ns, mtime_ns), follow_symlinks=False)
        os.replace(temp_path, target)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def extract_one_shard(
    *,
    tar_path: Path,
    expected_checksum: str | None,
    expected_members: list[ExpectedMember],
    destination: Path,
    temp_root: Path,
    progress: ProgressReporter,
    verify_checksum: bool,
) -> None:
    expected_index = 0

    try:
        raw = tar_path.open("rb")
    except OSError as exc:
        raise ExtractionError(f"Could not open shard {tar_path}: {exc}") from exc

    reader = HashingReader(raw)
    try:
        try:
            archive = tarfile.open(fileobj=reader, mode="r|")
        except (tarfile.TarError, OSError) as exc:
            raise ExtractionError(f"Could not open tar {tar_path}: {exc}") from exc

        with archive:
            for info in archive:
                if expected_index >= len(expected_members):
                    raise ExtractionError(
                        f"Unexpected extra tar member in {tar_path.name}: {info.name}"
                    )

                expected = expected_members[expected_index]
                actual_name = info.name.rstrip("/")
                actual_kind = tar_member_kind(info)
                actual_size = info.size if actual_kind == "file" else 0

                actual_tuple = (actual_name, actual_kind, actual_size)
                expected_tuple = (
                    expected.archive_path,
                    expected.kind,
                    expected.size,
                )
                if actual_tuple != expected_tuple:
                    raise ExtractionError(
                        f"Manifest/tar mismatch in {tar_path.name} at member "
                        f"{expected_index + 1}:\n"
                        f"  expected {expected_tuple!r}\n"
                        f"  found    {actual_tuple!r}"
                    )

                target = safe_target(destination, expected.relative_path)
                progress.set_path(expected.relative_path or ".")

                if actual_kind == "directory":
                    target.mkdir(parents=True, exist_ok=True)
                    os.chmod(target, info.mode & 0o7777)
                elif actual_kind == "file":
                    fileobj = archive.extractfile(info)
                    if fileobj is None:
                        raise ExtractionError(
                            f"Could not read file member {info.name} in {tar_path.name}"
                        )
                    with fileobj:
                        write_file_atomically(
                            source=fileobj,
                            target=target,
                            temp_root=temp_root,
                            relative_path=expected.relative_path,
                            expected_size=expected.size,
                            mode=info.mode,
                            mtime_ns=expected.mtime_ns,
                            progress=progress,
                        )
                else:
                    raise ExtractionError(
                        f"Unsupported tar member type in {tar_path.name}: {info.name}"
                    )

                expected_index += 1

        if expected_index != len(expected_members):
            missing = expected_members[expected_index]
            raise ExtractionError(
                f"Shard ended early: {tar_path.name}; first missing member is "
                f"{missing.archive_path}"
            )

        # tarfile may stop after the end-of-archive blocks. Drain any physical
        # padding/trailing bytes so the SHA-256 covers the complete .tar file.
        while reader.read(BUFFER_SIZE):
            pass

        if verify_checksum:
            if expected_checksum is None:
                raise ExtractionError(
                    f"No SHA-256 entry found for {tar_path.name}"
                )
            actual_checksum = reader.hasher.hexdigest()
            if actual_checksum.lower() != expected_checksum.lower():
                raise ExtractionError(
                    f"SHA-256 mismatch after extracting {tar_path.name}:\n"
                    f"  expected {expected_checksum}\n"
                    f"  actual   {actual_checksum}"
                )
    except (tarfile.TarError, OSError) as exc:
        raise ExtractionError(f"Failed while extracting {tar_path}: {exc}") from exc
    finally:
        reader.close()


def restore_directory_metadata(
    destination: Path,
    members: list[ExpectedMember],
) -> None:
    directories = [member for member in members if member.kind == "directory"]
    # Deepest first so changing a child does not subsequently alter its parent.
    directories.sort(
        key=lambda member: len(PurePosixPath(member.relative_path).parts),
        reverse=True,
    )
    for member in directories:
        target = safe_target(destination, member.relative_path)
        if not target.is_dir():
            raise ExtractionError(
                f"Expected extracted directory is missing before finalization: {target}"
            )
        if member.mtime_ns is not None:
            os.utime(target, ns=(member.mtime_ns, member.mtime_ns), follow_symlinks=False)


def validate_preflight(
    *,
    shard_dir: Path,
    destination: Path,
    plan: dict,
    shards: list[dict],
    members: list[ExpectedMember],
    manifest_digest: str,
    by_shard: dict[int, list[ExpectedMember]],
    checksums: dict[str, str],
    verify_checksum: bool,
) -> None:
    if plan.get("format") != "uncompressed-pax-tar-shards":
        raise ExtractionError(
            f"Unsupported shard format: {plan.get('format')!r}"
        )
    if plan.get("manifest_sha256") != manifest_digest:
        raise ExtractionError("manifest.jsonl does not match shard_plan.json")
    if int(plan.get("total_shards", -1)) != len(shards):
        raise ExtractionError("total_shards does not match the shard records")
    if int(plan.get("total_members", -1)) != len(members):
        raise ExtractionError("total_members does not match manifest.jsonl")

    expected_total_payload = sum(
        member.size for member in members if member.kind == "file"
    )
    if int(plan.get("total_payload_bytes", -1)) != expected_total_payload:
        raise ExtractionError("total_payload_bytes does not match manifest.jsonl")

    shard_dir_resolved = shard_dir.resolve()
    destination_resolved = destination.resolve(strict=False)
    if destination_resolved == shard_dir_resolved or shard_dir_resolved in destination_resolved.parents:
        raise ExtractionError(
            "The destination may not be the shard directory or a directory inside it."
        )

    for record in shards:
        index = int(record["index"])
        filename = str(record["filename"])
        tar_path = shard_dir / filename
        if not tar_path.is_file():
            raise ExtractionError(f"Planned shard is missing: {tar_path}")
        expected = by_shard.get(index, [])
        if len(expected) != int(record["member_count"]):
            raise ExtractionError(
                f"Manifest member count does not match plan for {filename}"
            )
        payload = sum(item.size for item in expected if item.kind == "file")
        if payload != int(record["payload_bytes"]):
            raise ExtractionError(
                f"Manifest payload size does not match plan for {filename}"
            )
        if verify_checksum and filename not in checksums:
            raise ExtractionError(f"SHA256SUMS has no entry for {filename}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract tar shards into an exact destination dataset directory, "
            "with restartable progress and in-stream checksum verification."
        )
    )
    parser.add_argument(
        "shard_directory",
        type=Path,
        help="Directory containing shard_plan.json, manifest.jsonl, and the .tar shards",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help=(
            "Exact directory to create as the restored dataset; it may have a "
            "different name from the original source directory"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds between progress updates (default: 1)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Override the external resume-state path",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help=(
            "Do not compare each tar against SHA256SUMS while extracting. "
            "Manifest member validation still occurs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow extraction into a nonempty destination when no matching resume "
            "state exists. Matching files may be overwritten; unrelated files are "
            "left in place."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shard_dir = args.shard_directory.expanduser().resolve()
    destination = args.destination.expanduser().resolve(strict=False)
    state_path = (
        args.state_file.expanduser().resolve(strict=False)
        if args.state_file is not None
        else default_state_path(destination)
    )
    temp_root = default_temp_root(destination)
    verify_checksum = not args.no_checksum

    if not shard_dir.is_dir():
        raise ExtractionError(f"Shard directory does not exist: {shard_dir}")

    plan = load_json(shard_dir / "shard_plan.json")
    shards = planned_shards(plan)
    members, manifest_digest, by_shard = load_manifest(
        shard_dir / "manifest.jsonl"
    )
    checksums = (
        load_sha256sums(shard_dir / "SHA256SUMS")
        if verify_checksum
        else {}
    )

    validate_preflight(
        shard_dir=shard_dir,
        destination=destination,
        plan=plan,
        shards=shards,
        members=members,
        manifest_digest=manifest_digest,
        by_shard=by_shard,
        checksums=checksums,
        verify_checksum=verify_checksum,
    )

    state = initialize_or_load_state(
        state_path=state_path,
        destination=destination,
        plan_id=str(plan["plan_id"]),
        shard_count=len(shards),
        force=args.force,
    )
    destination.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    completed_set = set(int(item) for item in state["completed_shards"])
    valid_indices = {int(record["index"]) for record in shards}
    unknown = completed_set - valid_indices
    if unknown:
        raise ExtractionError(
            f"Resume state contains unknown shard indices: {sorted(unknown)}"
        )

    completed_payload = sum(
        int(record["payload_bytes"])
        for record in shards
        if int(record["index"]) in completed_set
    )
    total_payload = int(plan["total_payload_bytes"])
    remaining = [
        record for record in shards if int(record["index"]) not in completed_set
    ]

    print(f"Shard directory: {shard_dir}")
    print(f"Destination:     {destination}")
    print(f"State file:      {state_path}")
    print(f"Shards:          {len(shards):,}")
    print(f"Payload:         {human_bytes(total_payload)}")
    print(
        f"Resume status:   {len(completed_set):,}/{len(shards):,} shards complete "
        f"({human_bytes(completed_payload)})"
    )
    print(
        "Checksum:        "
        + ("SHA-256 while extracting" if verify_checksum else "disabled")
    )

    free_bytes = shutil.disk_usage(destination.parent).free
    remaining_payload = total_payload - completed_payload
    safety_bytes = max(1024**3, remaining_payload // 100)
    required_bytes = remaining_payload + safety_bytes
    print(f"Free space:      {human_bytes(free_bytes)}")
    print(f"Estimated need:  {human_bytes(required_bytes)}")
    if free_bytes < required_bytes:
        raise ExtractionError(
            f"Insufficient free space: need about {human_bytes(required_bytes)}, "
            f"but only {human_bytes(free_bytes)} is available"
        )

    resume_base = completed_payload
    overall_started = time.monotonic()

    for record in remaining:
        index = int(record["index"])
        filename = str(record["filename"])
        payload_bytes = int(record["payload_bytes"])
        tar_path = shard_dir / filename

        print(
            f"\n[{index}/{len(shards)}] Extracting {filename} "
            f"({human_bytes(payload_bytes)})"
        )
        progress = ProgressReporter(
            shard_number=index,
            shard_count=len(shards),
            shard_total=payload_bytes,
            overall_base=completed_payload,
            overall_resume_base=resume_base,
            overall_total=total_payload,
            overall_started=overall_started,
            interval=args.progress_interval,
        )

        extract_one_shard(
            tar_path=tar_path,
            expected_checksum=checksums.get(filename),
            expected_members=by_shard.get(index, []),
            destination=destination,
            temp_root=temp_root,
            progress=progress,
            verify_checksum=verify_checksum,
        )
        progress.finish()

        completed_set.add(index)
        completed_payload += payload_bytes
        state["completed_shards"] = sorted(completed_set)
        state["updated_unix"] = time.time()
        atomic_write_json(state_path, state)
        print(f"Completed {filename}")

    print("\nRestoring directory modification times...")
    restore_directory_metadata(destination, members)

    state["completed_shards"] = sorted(completed_set)
    state["complete"] = True
    state["completed_unix"] = time.time()
    state["updated_unix"] = time.time()
    atomic_write_json(state_path, state)

    try:
        shutil.rmtree(temp_root)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"Warning: could not remove temporary directory {temp_root}: {exc}")

    elapsed = time.monotonic() - overall_started
    print("\nExtraction complete.")
    print(f"Restored dataset: {destination}")
    print(f"Payload written:  {human_bytes(total_payload)}")
    print(f"Elapsed:          {human_duration(elapsed)}")
    print(f"Resume record:    {state_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
