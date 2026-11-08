"""Static full cost of centroid keying, sorting, scalar queries and cache rebuild."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from warpower import SurfaceDomain
from warpower import _surface_query_cache as original
from surface_query_order_v2 import OrderedSurfaceQueryCache

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
with np.load(args.input) as stored:
    input_points = stored["input_points"]
    input_faces = stored["input_faces"]
    sites = stored["points"]
    cells = stored["full_tetrahedra"]
domain = SurfaceDomain(input_points, input_faces, surface_subdivisions=1, crease_subdivisions=1, feature_angle=120)
assert domain.error.numpy()[0] == 0
capacity = len(cells) + 37
padded = np.full((capacity, 4), -7, dtype=np.int32)
padded[:len(cells)] = cells
points = wp.array(sites, dtype=wp.vec3, device=domain.device)
tets = wp.array(padded, dtype=int, device=domain.device)
count = wp.full(1, len(cells), dtype=int, device=domain.device)
caches = {"before": original.SurfaceQueryCache(domain, capacity), "after": OrderedSurfaceQueryCache(domain, capacity)}
report = dict(scope=__doc__, limits="Immutable validated final meshes only; no full-solve or invalid-input qualification. Warm rows repeatedly query unchanged input; cold rows include clearing the cache. Both include cache rebuild. Setup and host validation are outside timing.", source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), sites=len(sites), cells=len(cells), capacity=capacity, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("surface_query_order_v1.py"), Path(__file__).with_name("surface_query_order_v2.py"), Path("warpower/_surface_query_cache.py")]}, rows=[])
start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
for mode, kernel in [("full", original.classify_cached), ("selection", original.selection_cached), ("retained", original.retained_cached)]:
    outputs = {}
    graphs = {}
    for phase, cache in caches.items():
        error = wp.zeros(1, dtype=int, device=domain.device)
        keep = wp.full(capacity, 77, dtype=int, device=domain.device)
        quality = wp.full(capacity, 77, dtype=wp.float64, device=domain.device)
        arrays = dict(error=error, keep=keep, quality=quality)
        inputs = [domain.tree, points, tets, count, 0.4, error]
        if mode == "retained":
            inputs += [keep, quality]
        else:
            center = wp.full(capacity, wp.vec3d(77), dtype=wp.vec3d, device=domain.device)
            radius = wp.full(capacity, 77, dtype=wp.float64, device=domain.device)
            distance = wp.full(capacity, 77, dtype=wp.float64, device=domain.device)
            volume = wp.full(capacity, 77, dtype=wp.float64, device=domain.device)
            arrays.update(center=center, radius=radius, distance=distance, volume=volume)
            inputs += [center, radius, distance, keep, volume, quality]
        outputs[phase] = arrays
        for temperature in ["cold", "warm"]:
            with wp.ScopedCapture(device=domain.device) as cap:
                error.zero_()
                if temperature == "cold":
                    cache.reset()
                cache.launch(kernel, capacity, inputs)
            graphs[(phase, temperature)] = cap.graph
    for temperature in ["cold", "warm"]:
        reference = None
        for phase in ["before", "after"]:
            graph = graphs[(phase, temperature)]
            for _ in range(3):
                wp.capture_launch(graph)
            wp.synchronize_device(domain.device)
            before = cpu_quota_snapshot()
            samples = []
            for _ in range(20):
                wp.record_event(start)
                wp.capture_launch(graph)
                wp.record_event(end)
                wp.synchronize_event(end)
                samples.append(wp.get_event_elapsed_time(start, end))
            quota = cpu_quota_delta(before, cpu_quota_snapshot())
            values = {name: array.numpy() for name, array in outputs[phase].items()}
            assert values["error"][0] == 0
            if reference is None:
                reference = values
            else:
                for name, data in values.items():
                    assert data.tobytes() == reference[name].tobytes(), (mode, temperature, name)
                order = caches["after"].order_ids.numpy()[:capacity]
                np.testing.assert_array_equal(np.sort(order), np.arange(capacity))
            row = dict(mode=mode, cache=temperature, phase=phase, gpu_ms=samples, median_ms=float(np.median(samples)), cpu_quota_delta=quota, quota_clean=all(v["throttled_periods"] == 0 for v in quota.values()), error=0, exact_outputs_match=phase == "after", output_sha256={name: hashlib.sha256(data.tobytes()).hexdigest() for name, data in values.items()})
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(mode, temperature, phase, row["median_ms"], flush=True)
            assert row["quota_clean"], row
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
