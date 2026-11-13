#!/usr/bin/env python3
"""Download I_512_1850_UnifWMAP_Wide shards anonymously from public Box links.

Requires rclone. No Box account, rclone remote, or Box credentials required.
Archives are verified with SHA-256 when downloaded. Re-running skips complete
files. Interrupted transfers may restart the current file (not partial bytes).

This is intentionally separate from the original CMB-ML ICCV get_data code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DEFAULT_MANIFEST = Path(__file__).with_name('UnifWMAP_Wide_download_manifest.json')
VALID_ID = re.compile(r'^[0-9]+$')
VALID_TOKEN = re.compile(r'^[A-Za-z0-9]+$')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_existing(path: Path, item: dict, verify_hash: bool = True) -> bool:
    if not path.is_file() or path.stat().st_size != item['size_bytes']:
        return False
    expected = item.get('sha256')
    return not (verify_hash and expected) or digest(path) == expected


def download(item: dict, *, dest: Path, url_template: str,
             token: str, verify_existing: bool, dry_run: bool,
             progress: bool) -> tuple[str, str]:
    name = item['name']
    target = dest / name
    if valid_existing(target, item, verify_existing):
        return name, 'already present'
    if dry_run:
        return name, 'would download'

    # Use a temporary file; publish a shard only after size and hash match.
    temp = dest / (name + '.partial')
    if temp.exists():
        temp.unlink()  # rclone copyurl is not a byte-range resume mechanism.
    url = url_template.format(folder_shared_token=token, box_id=item['box_id'])
    command = ['rclone', 'copyurl', url, str(temp), '--retries', '3',
               '--low-level-retries', '10']
    if progress:
        command.append('--progress')
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'rclone copyurl returned {result.returncode}: {name}')
    if temp.stat().st_size != item['size_bytes']:
        raise RuntimeError(f'Wrong byte count for {name}: {temp.stat().st_size} != {item["size_bytes"]}')
    expected = item.get('sha256')
    if expected and digest(temp) != expected:
        raise RuntimeError(f'SHA-256 mismatch: {name}')
    os.replace(temp, target)
    return name, 'downloaded and checked' if expected else 'downloaded (size checked)'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='JSON manifest generated for the published dataset')
    parser.add_argument('--dest', type=Path, default=Path('UnifWMAP_Wide_shards'),
                        help='Download destination; defaults to ./UnifWMAP_Wide_shards')
    parser.add_argument('--only', action='append', default=[], metavar='FILENAME',
                        help='Exact filename to download; repeat for multiple files')
    parser.add_argument('--include-extras', action='store_true',
                        help='Also download auxiliary inventory/shard-info files')
    parser.add_argument('--jobs', type=int, default=2, help='Concurrent downloads (default: 2)')
    parser.add_argument('--trust-existing-sizes', action='store_true',
                        help='Skip hashing existing files (newly downloaded files are still hash-checked)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--progress', action='store_true', help='Show rclone transfer progress')
    args = parser.parse_args()

    if args.jobs < 1 or args.jobs > 8:
        parser.error('--jobs must be between 1 and 8')
    if not args.dry_run and shutil.which('rclone') is None:
        parser.error('rclone is not installed or not on PATH')
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema_version'] != 1:
        parser.error('Unsupported manifest schema version')
    token = manifest['folder_shared_token']
    if not VALID_TOKEN.fullmatch(token):
        parser.error('Invalid Box sharing token')
    items = manifest['files']
    names = set()
    for item in items:
        name = item['name']
        if not name or name != Path(name).name or '/' in name or '\\' in name or name in names:
            parser.error(f'Unsafe or duplicate filename: {name!r}')
        names.add(name)
        if not VALID_ID.fullmatch(item['box_id']):
            parser.error(f'Invalid Box file ID in record {name!r}')
        if not isinstance(item.get('size_bytes'), int) or item['size_bytes'] < 0:
            parser.error(f'Invalid byte count for {name!r}')

    if args.only:
        missing = set(args.only) - names
        if missing:
            parser.error(f'Unknown file(s): {sorted(missing)}')
        selected = [x for x in items if x['name'] in args.only]
    else:
        selected = [x for x in items if args.include_extras or x['kind'] != 'extra']

    args.dest.mkdir(parents=True, exist_ok=True)
    total = sum(item['size_bytes'] for item in selected)
    print(f'{len(selected)} files selected; {total:,} bytes; destination: {args.dest}', flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download, x, dest=args.dest,
                               url_template=manifest['download_url_pattern'],
                               token=token, verify_existing=not args.trust_existing_sizes,
                               dry_run=args.dry_run, progress=args.progress): x['name'] for x in selected}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, status = future.result()
                print(f'{status}: {name}', flush=True)
            except Exception as exc:
                errors.append((name, str(exc)))
                print(f'FAILED: {name}: {exc}', file=sys.stderr, flush=True)
    if errors:
        print(f'{len(errors)} file(s) failed. Re-run to retry.', file=sys.stderr)
        return 1
    print('Selected files complete. Archives with published hashes were checked on download.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
