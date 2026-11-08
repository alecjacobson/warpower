"""One device-controlled producer/large-mesher/consumer graph with failures and recovery."""

import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.mesher import surface
from tests.test_vtm_core import consume
from warpower import SurfaceDomain, VariationalMeshingWorkspace


@wp.kernel
def produce(
    source: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    phase: wp.array[int],
    target: wp.array[int],
    seed: wp.array[int],
):
    i = wp.tid()
    r = phase[0]
    p = source[i]
    if r == 1:
        p = wp.vec3(p[0] * 1.125, p[1], p[2])
    if r == 3:
        p = wp.vec3(0.0)
    points[i] = p
    if i == 0:
        target[0] = 4096
        if r == 1:
            target[0] = 8192
        if r == 2:
            target[0] = 16385
        seed[0] = 2026


@wp.kernel
def save_points(
    phase: wp.array[int],
    count: wp.array[int],
    points: wp.array[wp.vec3],
    roles: wp.array[int],
    saved: wp.array2d[wp.vec3],
    saved_roles: wp.array2d[int],
):
    i = wp.tid()
    r = phase[0]
    if i < count[0]:
        saved[r, i] = points[i]
        saved_roles[r, i] = roles[i]


@wp.kernel
def save_cells(
    phase: wp.array[int], count: wp.array[int], cells: wp.array2d[int], saved: wp.array3d[int]
):
    i = wp.tid()
    if i < count[0]:
        for k in range(cells.shape[1]):
            saved[phase[0], i, k] = cells[i, k]


@wp.kernel
def finish(
    phase: wp.array[int],
    condition: wp.array[int],
    error: wp.array[int],
    valid: wp.array[int],
    counts: wp.array[int],
    topology: wp.array[int],
    volume: wp.array[wp.float64],
    reported_volume: wp.array[wp.float64],
    rows: wp.array2d[int],
    volumes: wp.array2d[wp.float64],
):
    r = phase[0]
    rows[r, 0] = error[0]
    rows[r, 1] = valid[0]
    for k in range(4):
        rows[r, k + 2] = counts[k]
        rows[r, k + 6] = topology[k]
    volumes[r, 0] = volume[0]
    volumes[r, 1] = reported_volume[0]
    phase[0] = r + 1
    condition[0] = int(r < 4)


p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dot", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
inputs = surface("torus")
domain = SurfaceDomain(*inputs, surface_subdivisions=4, crease_subdivisions=4)
source = wp.clone(domain.points)
target = wp.full(1, 4096, dtype=int)
seed = wp.full(1, 2026, dtype=int)
mesh = VariationalMeshingWorkspace(
    domain, 16384, target_count=target, initial_count=256, seed=seed, quality_rounds=4
)
phase = wp.zeros(1, dtype=int)
condition = wp.ones(1, dtype=int)
rows = wp.zeros((5, 10), dtype=int)
volumes = wp.zeros((5, 2), dtype=wp.float64)
volume = wp.zeros(1, dtype=wp.float64)
points = wp.empty((5, 16384), dtype=wp.vec3)
roles = wp.empty((5, 16384), dtype=int)
full = wp.empty((5, len(mesh.full_tetrahedra), 4), dtype=int)
material = wp.empty((5, len(mesh.tetrahedra), 4), dtype=int)
boundary = wp.empty((5, len(mesh.boundary_faces), 3), dtype=int)
wp.load_module(module=__name__, device=domain.device)
wp.load_module(module="tests.test_vtm_core", device=domain.device)
pointers = [
    mesh.points.ptr,
    mesh.full_tetrahedra.ptr,
    mesh.tetrahedra.ptr,
    mesh.boundary_faces.ptr,
    mesh.counts.ptr,
]


def step():
    wp.launch(produce, len(source), inputs=[source, domain.points, phase, target, seed])
    domain.refit()
    mesh.run()
    volume.zero_()
    wp.launch(
        consume,
        len(mesh.tetrahedra),
        inputs=[mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error, volume],
    )
    wp.launch(
        save_points,
        16384,
        inputs=[phase, mesh.point_count, mesh.points, mesh.core.boundary.roles, points, roles],
    )
    for count, cells, saved in [
        (mesh.full_tetrahedra_count, mesh.full_tetrahedra, full),
        (mesh.tetrahedra_count, mesh.tetrahedra, material),
        (mesh.boundary_count, mesh.boundary_faces, boundary),
    ]:
        wp.launch(save_cells, len(cells), inputs=[phase, count, cells, saved])
    wp.launch(
        finish,
        1,
        inputs=[
            phase,
            condition,
            mesh.error,
            mesh.valid,
            mesh.counts,
            mesh.topology,
            volume,
            mesh.volume,
            rows,
            volumes,
        ],
    )


with wp.ScopedCapture() as capture:
    phase.zero_()
    condition.fill_(1)
    wp.capture_while(condition, step)
wp.synchronize()
wp.capture_debug_dot_print(capture.graph, str(a.dot), verbose=True)
import torch

cuda = torch.cuda.cudart()
cuda.cudaProfilerStart()
wp.capture_launch(capture.graph)
wp.synchronize()
cuda.cudaProfilerStop()
values = rows.numpy()
consumed = volumes.numpy()
r = dict(
    scope=__doc__
    + " First launch; setup/readback outside trace, one external final synchronization. Diagnostic, not clean timing.",
    source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    columns=[
        "error",
        "valid",
        "sites",
        "full_tets",
        "material_tets",
        "boundary_faces",
        "components",
        "shells",
        "cavities",
        "Euler",
    ],
    phases=["valid4096", "deformed8192", "invalid16385", "collapsed_domain", "recovered4096"],
    metadata=values.tolist(),
    volumes=consumed.tolist(),
    stable_pointers=pointers
    == [
        mesh.points.ptr,
        mesh.full_tetrahedra.ptr,
        mesh.tetrahedra.ptr,
        mesh.boundary_faces.ptr,
        mesh.counts.ptr,
    ],
)
a.output.write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r), flush=True)
assert values[:, 0].tolist() == [0, 0, 1, 1, 0] and values[:, 1].tolist() == [1, 1, 0, 0, 1]
assert (
    values[:, 2].tolist() == [4096, 8192, 0, 0, 4096]
    and not values[2:4, 2:].any()
    and not consumed[2:4].any()
)
assert r["stable_pointers"] and np.all(consumed[[0, 1, 4]] > 0)
np.testing.assert_allclose(consumed[:, 0], consumed[:, 1], rtol=1e-11)
pv, rv, fv, mv, bv = points.numpy(), roles.numpy(), full.numpy(), material.numpy(), boundary.numpy()
for i in [0, 1, 4]:
    n, nf, nm, nb = values[i, 2:6]
    ip = inputs[0].copy()
    if i == 1:
        ip[:, 0] *= np.float32(1.125)
    path = a.output.with_name(a.output.stem + f"-frame{i}.npz")
    assert not path.exists()
    np.savez_compressed(
        path,
        points=pv[i, :n],
        roles=rv[i, :n],
        full_tetrahedra=fv[i, :nf],
        tetrahedra=mv[i, :nm],
        boundary=bv[i, :nb],
        input_points=ip,
        input_faces=inputs[1],
    )
