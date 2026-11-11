"""Exhaustive and independent rational oracles for changing bounded face queries."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_predicates import determinant, exact_reference
from tests.test_support_reuse import oriented_face
from warpower import _canonical
from warpower import _filtered_predicates as fp
from warpower._delaunay_certificate import symbolic_sphere
from warpower._facet_completion import extend_faces
from warpower._mesh_topology import face_vertices
from warpower._proved_completion import search_unproved_faces as analytic_search
from warpower._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


# Independent production baseline: the prior exact/ball search, without analytic filters.
@wp.kernel
def reference_search(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    error: wp.array[int],
    supported: wp.array[int],
    statistics: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // workers, tid % workers
    candidates[tid] = -1
    if i >= count[0]:
        return
    if lane == 0:
        wp.atomic_add(statistics, wp.where(supported[i] != 0, 0, 1), 1)
    if supported[i] != 0:
        return
    face = face_vertices(tets, faces[i])
    a, b, c = points[face[0]], points[face[1]], points[face[2]]
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        orientation = fp.orient3d(a, b, c, points[q])
        if orientation.valid == 0:
            wp.atomic_or(error, 0, 2)
            return
        if orientation.sign <= 0:
            continue
        if best < 0:
            best = q
        else:
            result = symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if result.sign > 0:
                best = q
    candidates[tid] = best


@wp.kernel
def produce(
    frames: wp.array2d[wp.vec3],
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    count: wp.array[int],
    faces: int,
):
    i = wp.tid()
    points[i] = frames[phase[0], i]
    if i == 0:
        count[0] = faces
        if phase[0] == 2:
            count[0] = faces // 2
        elif phase[0] == 4:
            count[0] = 0


def winner(points, face):
    owners = np.unique(points, axis=0, return_index=True)[1]
    weights = []
    for i in range(len(points)):
        v = i + 1
        v = ((v ^ (v >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        v = ((v ^ (v >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        v ^= v >> 16
        weights.append(Fraction(v & 0xFFFFFF, 1 << 23) - 1)
    best = -1
    for q in owners:
        if q in face:
            continue
        orientation = -determinant([[*map(float, points[i]), 1] for i in [*face, q]])
        if orientation <= 0:
            continue
        if best < 0:
            best = int(q)
            continue
        ids = [*face, best, int(q)]
        rows = points[ids]
        _, sign = exact_reference(rows, np.zeros(5))
        if sign == 0:
            _, sign = exact_reference(rows, [weights[i] for i in ids])
        if sign == 0:
            for site in sorted(ids):
                _, sign = exact_reference(rows, [int(i == site) for i in ids])
                if sign:
                    break
        assert sign != 0
        if sign > 0:
            best = int(q)
    return None if best < 0 else np.sort([*face, best])


@pytest.mark.parametrize(
    "case", ["cube", "octahedron", "thin", "random33", "random257", "torus4096"]
)
@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("block", [32, 256])
def test_changed_inputs_match_exhaustive_and_rational_winners(
    case, exponent, block, search_override=None
):
    rng = np.random.default_rng(164)
    if case == "torus4096":
        saved = np.load(
            "results/validation/vtm/large-fragment-reuse/large-reuse-final/torus4096-after.npz"
        )
        p = saved["points"]
        t = saved["full_tetrahedra"]
    else:
        p = np.array(list(product([-1, 1], repeat=3)), np.float32)
        if case == "octahedron":
            p = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
        if case == "thin":
            p[:, 2] *= np.float32(2**-30)
        if case.startswith("random"):
            p = rng.normal(size=(int(case[6:]), 3)).astype(np.float32)
        # Compute candidates before anisotropic scaling, avoiding Qhull's thin-cell tolerance.
        query = p.copy()
        if case == "thin":
            query[:, 2] *= np.float32(2**30)
        t = Delaunay(query).simplices
        rows = []
        for ids in t:
            sign = -determinant([[*map(float, p[i]), 1] for i in ids])
            if sign:
                rows.append(ids if sign > 0 else ids[[1, 0, 2, 3]])
        t = np.asarray(rows, np.int32)
    t = t[np.linspace(0, len(t) - 1, min(32, len(t)), dtype=int)].astype(np.int32)
    p = np.vstack([p, p[:3]])
    scale = np.float32(2.0**exponent)
    p = p * scale
    moved = p * np.float32(2) + np.array([4, 7, -5], np.float32) * scale
    moved[5] += np.array([0.125, -0.25, 0.5], np.float32) * scale
    permuted = np.roll(p, 3, axis=0)
    unrelated = rng.normal(size=p.shape).astype(np.float32) * scale
    unrelated[-3:] = unrelated[:3]
    frames = np.stack([p, moved, permuted, unrelated, unrelated, p])
    faces = np.concatenate(
        [np.arange(4 * len(t), dtype=np.int32), np.array([0, 1, 0, 3], np.int32)]
    )
    rng.shuffle(faces)
    device = "cuda:0"
    n = len(p)
    size = len(faces) + 7
    source = wp.array(frames, dtype=wp.vec3, device=device)
    points = wp.array(p, dtype=wp.vec3, device=device)
    tets = wp.array(t, dtype=int, device=device)
    indices = wp.array(faces, dtype=int, device=device)
    phase = wp.zeros(1, dtype=int, device=device)
    count = wp.full(1, len(faces), dtype=int, device=device)
    weights = wp.zeros(n, dtype=float, device=device)
    owners, retained = _canonical.canonicalize(points, weights)
    table, buckets, bits = retained
    hierarchy = SiteHierarchy(points)
    hierarchy.tree.owners = owners
    errors = wp.zeros(2, dtype=int, device=device)
    supported = wp.zeros(len(faces), dtype=int, device=device)
    statistics = wp.zeros(2, dtype=int, device=device)
    candidates = [wp.empty(32 * size, dtype=int, device=device) for _ in range(2)]
    outputs = [wp.empty(size, dtype=wp.vec4i, device=device) for _ in range(2)]
    live = [wp.empty(size, dtype=int, device=device) for _ in range(2)]
    for module in [__name__, "warpower._facet_completion", "warpower._proved_completion"]:
        wp.load_module(module=module, device=device, block_dim=block)
    wp.load_module(module=__name__, device=device)
    if search_override is not None:
        search_override.prepare(candidates[1], device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, n, inputs=[source, phase, points, count, len(faces)], device=device)
        table.fill_(-1)
        wp.launch(
            _canonical.insert, n, inputs=[points, weights, bits, table, buckets], device=device
        )
        wp.launch(_canonical.resolve, n, inputs=[table, buckets, owners], device=device)
        hierarchy.rebuild()
        errors.zero_()
        statistics.zero_()
        for j, kernel in enumerate([reference_search, analytic_search]):
            tree = owners
            inputs = [
                points,
                tree,
                tets,
                indices,
                count,
                candidates[j],
                32,
                errors[j : j + 1],
                supported,
                statistics,
            ]
            if j == 1 and search_override is not None:
                search_override.launch(inputs, block)
            else:
                wp.launch(kernel, 32 * size, inputs=inputs, device=device, block_dim=block)
            wp.launch(
                extend_faces,
                size,
                inputs=[
                    points,
                    tets,
                    indices,
                    count,
                    candidates[j],
                    32,
                    outputs[j],
                    live[j],
                    0,
                    errors[j : j + 1],
                ],
                device=device,
                block_dim=block,
            )
    for value in range(len(frames)):
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(errors.numpy(), [0, 0])
        # Both searches use the same strided partition: every lane must agree.
        np.testing.assert_array_equal(candidates[0].numpy(), candidates[1].numpy())
        a, b = [v.numpy() for v in live]
        np.testing.assert_array_equal(a, b)
        x, y = [v.numpy() for v in outputs]
        np.testing.assert_array_equal(x[a != 0], y[b != 0])
        active = len(faces) // 2 if value == 2 else 0 if value == 4 else len(faces)
        if search_override is not None:
            search_override.observe(value, active)
        for c in candidates:
            assert np.all(c.numpy().reshape(size, 32)[active:] == -1)
        if n <= 20:
            for slot in np.linspace(0, active - 1, min(6, active), dtype=int):
                want = winner(frames[value], oriented_face(t, faces[slot]))
                assert bool(b[slot]) == (want is not None)
                if want is not None:
                    np.testing.assert_array_equal(y[slot], want)
