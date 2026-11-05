"""Extract public RadFoam v1 geometry for the scenes benchmarked by Paragram.

Install the ``datasets`` extra, then run ``python -m benchmarks.fetch_scenes``.
Only positions are retained; checkpoint hashes and public IDs are recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENES = {
    "bicycle": "13e_DP83_bE_sI3D4iR8YDhomlb9PenYR",
    "bonsai": "1QPT5soWRQ4JT8KGPrbT7mmviCVpYzsu9",
    "counter": "15loqFx9RI8rcDv6G4YLuWczW8557aR5i",
    "drjohnson": "1xorGfdc1nlgRZ-iJAi48_Wo5Rl2mShq9",
    "garden": "1lLEIfH7CBLAEmdqJrDmQ6flK5Qub7ttQ",
    "kitchen": "1nXqHnN6Wjo1EmQ7jX5PRrqH3OjU5M3xX",
    "playroom": "1hIlFwJmby7k0Mc_QqscH_w9BQCkfezmB",
    "room": "1eBr8SKYZZwGkPsHO_2Kgebyzd60_HN5v",
    "stump": "1kLtVZrR9xSjAyL-pL4s4wgbOwW6qVydk",
}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    import gdown
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--output", type=Path, default=Path("data/radfoam"))
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        output = args.output / f"{scene}.npz"
        manifest = args.output / f"{scene}.json"
        if output.exists() and manifest.exists():
            print(f"{scene}: already extracted", flush=True)
            continue
        checkpoint = args.output / f"{scene}.pt"
        if not checkpoint.exists():
            gdown.download(id=SCENES[scene], output=str(checkpoint), quiet=True)
        checksum = file_hash(checkpoint)
        model = torch.load(checkpoint, map_location="cpu", weights_only=True)
        points = model["xyz"].detach().numpy().astype(np.float32)
        np.savez(output, points=points)
        record = {
            "scene": scene,
            "checkpoint_version": "RadFoam checkpoints_v1",
            "google_drive_id": SCENES[scene],
            "checkpoint_sha256": checksum,
            "n": len(points),
            "positions_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "npz_sha256": file_hash(output),
            "source": "https://github.com/theialab/radfoam#checkpoints",
        }
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        del model, points
        if not args.keep_checkpoint:
            checkpoint.unlink()
        print(f"{scene}: {record['n']} sites", flush=True)


if __name__ == "__main__":
    main()
