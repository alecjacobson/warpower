"""Prepare a pinned Stanford Armadillo scan; host I/O precedes device meshing."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from examples.vtm_inputs import write_off

COMMIT = "644dd4104843b6d736745d9dafbd70bf8d175648"
SOURCE = f"https://raw.githubusercontent.com/libigl/libigl-tutorial-data/{COMMIT}/armadillo.obj"
SHA256 = "c007ebd11e541a2b1ab86a4c376653579281265f2c6c786a4a7bf47ba84d6879"
TERMS = "https://graphics.stanford.edu/data/3Dscanrep/"


def prepare(output, source=None):
    """Verify the original bytes, center/scale coordinates, and retain connectivity."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = output / "armadillo.obj"
    cached = Path(source) if source is not None else original
    if cached.exists():
        data = cached.read_bytes()
    else:
        if source is not None:
            raise FileNotFoundError(cached)
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Armadillo source SHA256 does not match the pinned distribution")
    vertices, triangles = [], []
    for line in data.decode("utf-8").splitlines():
        row = line.split()
        if not row or row[0].startswith("#"):
            continue
        if row[0] == "v":
            if len(row) != 4:
                raise ValueError("Expected three coordinates in the pinned OBJ")
            vertices.append([float(v) for v in row[1:]])
        elif row[0] == "f":
            if len(row) != 4:
                raise ValueError("Expected triangular faces in the pinned OBJ")
            triangles.append([int(v.split("/")[0]) - 1 for v in row[1:]])
    points, faces = np.asarray(vertices, np.float64), np.asarray(triangles, np.int32)
    if points.shape != (43243, 3) or faces.shape != (86482, 3):
        raise ValueError("Unexpected pinned mesh dimensions")
    if not np.isfinite(points).all() or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Invalid pinned OBJ coordinates or indices")
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    scale = float((upper - lower).max() / 2)
    normalized = ((points - center) / scale).astype(np.float32)
    original.write_bytes(data)
    off = output / "armadillo.off"
    write_off(off, normalized, faces)
    metadata = dict(
        model="Stanford Armadillo, reduced libigl tutorial-data distribution",
        attribution="Stanford University Computer Graphics Laboratory; libigl tutorial-data distribution",
        source=SOURCE,
        source_commit=COMMIT,
        source_sha256=SHA256,
        source_bytes=len(data),
        attribution_reference=f"https://raw.githubusercontent.com/libigl/libigl-tutorial-data/{COMMIT}/README.md",
        terms_reference=TERMS,
        terms_summary="Research use and free redistribution require source acknowledgment; commercial use requires Stanford permission. This asset is not relicensed under the Warpower code license.",
        vertices=len(points),
        triangles=len(faces),
        preprocessing=dict(
            operation="Center bounding box, divide by half largest extent, convert to float32; connectivity unchanged. No hole filling, snapping, simplification or repair.",
            center=center.tolist(),
            scale=scale,
        ),
        off_sha256=hashlib.sha256(off.read_bytes()).hexdigest(),
        validation_scope="This downloader verifies source identity and array format only. Device domain validation and independent mesh checks belong to the meshing experiment.",
        paper_relation="Additional scanned benchmark; Armadillo is not a model shown in the Alliez VTM paper.",
    )
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Optional local copy; the same pinned SHA256 is required"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.source), indent=2))


if __name__ == "__main__":
    main()
