"""Pack or verify checksum-pinned experiment bundles without importing Warp."""

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def members(document):
    result = {}
    for row in document["files"]:
        name = row["path"]
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
            or "\0" in name
            or name in result
        ):
            raise ValueError(f"Unsafe or duplicate member: {name!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError(f"Invalid byte count for {name}")
        result[name] = row
    return result


def pack(root, inventory, archive, manifest):
    root, archive, manifest = Path(root).resolve(), Path(archive), Path(manifest)
    rows = members(inventory)
    if archive.exists() or manifest.exists() or archive.resolve() == manifest.resolve():
        raise FileExistsError("Bundle and manifest must be new, distinct paths")
    # Check everything before producing an archive; a stale inventory is a failure.
    for name, row in rows.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes root or is a link: {name}")
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError(f"Source differs from inventory: {name}")
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name in sorted(rows):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = rows[name]["bytes"], 0o644
                    with (root / name).open("rb") as source:
                        tar.addfile(info, source)
    result = dict(inventory)
    result["archive"] = dict(
        name=archive.name, bytes=archive.stat().st_size, sha256=file_digest(archive)
    )
    verify(archive, result)
    with manifest.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    return result


def verify(archive, manifest):
    archive = Path(archive)
    expected = members(manifest)
    identity = manifest["archive"]
    if archive.stat().st_size != identity["bytes"] or file_digest(archive) != identity["sha256"]:
        raise ValueError("Archive size or SHA-256 differs from manifest")
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for entry in tar:
            if entry.name not in expected or entry.name in seen or not entry.isfile():
                raise ValueError(f"Unexpected, duplicate or non-regular member: {entry.name}")
            row = expected[entry.name]
            with tar.extractfile(entry) as source:
                if entry.size != row["bytes"] or digest(source) != row["sha256"]:
                    raise ValueError(f"Member differs from manifest: {entry.name}")
            seen.add(entry.name)
    if seen != set(expected):
        raise ValueError(f"Missing archive members: {sorted(set(expected) - seen)}")
    return dict(verified_members=len(seen), archive_sha256=identity["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("pack")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--inventory", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(
            args.root, json.loads(args.inventory.read_text()), args.archive, args.manifest
        )
        print(json.dumps(result["archive"]))
    else:
        print(json.dumps(verify(args.archive, json.loads(args.manifest.read_text()))))


if __name__ == "__main__":
    main()
