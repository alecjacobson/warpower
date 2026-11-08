"""Captured current-input equivalence of the private new-cell query prototype."""

from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest
import warp as wp
from prepared_new_cell_v2 import PreparedUnprovedQueries, prepare_unproved
from scipy.spatial import Delaunay

from tests.test_predicates import exact_reference
from tests.test_prepared_query_masks import positive_canonical
from warpower import _canonical
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, sphere, symbolic_sphere
from warpower._proved_completion import check_unproved_spheres
from warpower._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def produce(
    frames: wp.array2d[wp.vec3],
    cells: wp.array2d[wp.vec4i],
    counts: wp.array[int],
    proofs: wp.array2d[int],
    initial_errors: wp.array[int],
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    proved: wp.array[int],
    initial: wp.array[int],
):
    i = wp.tid()
    f = phase[0]
    if i < points.shape[0]:
        points[i] = frames[f, i]
    if i < tets.shape[0]:
        ids = cells[f, i]
        for k in range(4):
            tets[i, k] = ids[k]
        proved[i] = proofs[f, i]
    if i == 0:
        count[0] = counts[f]
        initial[0] = initial_errors[f]


@wp.kernel
def reset(
    initial: wp.array[int],
    error: wp.array[int],
    statistics: wp.array[int],
    proof_statistics: wp.array[int],
):
    error[0] = initial[0]
    statistics[0] = 7
    proof_statistics[0] = 11
    proof_statistics[1] = 13


@wp.kernel
def poison(shapes: wp.array[Sphere], boxes: wp.array[sb.Box]):
    i = wp.tid()
    shapes[i] = Sphere()
    boxes[i] = sb.Box()


@wp.kernel
def invalidate(boxes: wp.array[sb.Box]):
    i = wp.tid()
    box = boxes[i]
    box.valid = 0
    boxes[i] = box


@wp.kernel
def consume(
    phase: wp.array[int],
    which: int,
    error: wp.array[int],
    statistics: wp.array[int],
    proof_statistics: wp.array[int],
    output: wp.array3d[int],
):
    f = phase[0]
    output[f, which, 0] = error[0]
    output[f, which, 1] = statistics[0]
    output[f, which, 2] = proof_statistics[0]
    output[f, which, 3] = proof_statistics[1]


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


@wp.kernel
def reference_without_box(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    statistics: wp.array[int],
    proved: wp.array[int],
    proof_statistics: wp.array[int],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if i >= count[0]:
        return
    if proved[i] != 0:
        if lane == 0:
            wp.atomic_add(proof_statistics, 0, 1)
        return
    if lane == 0:
        wp.atomic_add(proof_statistics, 1, 1)
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = sphere(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
    box = sb.bounds(shape.origin, shape.nx, shape.ny, shape.nz, shape.determinant)
    box.valid = 0
    queries = int(0)
    for q in range(lane, points.shape[0], 32):
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        queries += 1
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
    wp.atomic_add(statistics, 0, queries)


def rational_error(points, cells, proved, initial):
    owners = np.unique(points, axis=0, return_index=True)[1]
    weights = []
    for i in range(len(points)):
        v = i + 1
        v = ((v ^ (v >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        v = ((v ^ (v >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        v ^= v >> 16
        weights.append(Fraction(v & 0xFFFFFF, 1 << 23) - 1)
    error = int(initial)
    for t, known in zip(cells, proved):
        if known:
            continue
        for q in owners:
            if q in t:
                continue
            ids = [*t, int(q)]
            p = points[ids]
            _, value = exact_reference(p, np.zeros(5))
            if value == 0:
                _, value = exact_reference(p, [weights[i] for i in ids])
            if value == 0:
                for site in sorted(ids):
                    _, value = exact_reference(p, [int(i == site) for i in ids])
                    if value:
                        break
            if value == 0:
                error |= 2
            elif value > 0:
                error |= 8
    return error


@pytest.mark.parametrize(
    "case", ["cube", "octahedron", "flat", "random33", "random4097", "torus16384"]
)
@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("block", [32, 256])
def test_captured_geometry_errors_queries_and_proofs_match(case, exponent, block):
    rng = np.random.default_rng(233001)
    if case == "cube":
        p = np.asarray(list(product([-1, 1], repeat=3)), np.float32)
    elif case == "octahedron":
        p = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
    elif case == "flat":
        p = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]], np.float32)
    elif case.startswith("random"):
        p = rng.normal(size=(int(case[6:]), 3)).astype(np.float32)
    else:
        n = 16384
        u, v = rng.uniform(0, 2 * np.pi, (2, n))
        r = 0.25 * np.sqrt(rng.uniform(0, 1, n))
        p = np.column_stack(
            ((1 + r * np.cos(v)) * np.cos(u), (1 + r * np.cos(v)) * np.sin(u), r * np.sin(v))
        ).astype(np.float32)
    if case == "flat":
        candidates = np.asarray([[0, 1, 2, 3]], np.int32)
    elif len(p) <= 8:
        candidates = np.asarray(list(combinations(range(len(p)), 4)), np.int32)
    else:
        full = Delaunay(p).simplices
        candidates = np.vstack(
            [
                full[np.linspace(0, len(full) - 1, 12, dtype=int)],
                np.asarray([rng.choice(len(p), 4, replace=False) for _ in range(12)], np.int32),
            ]
        )
    frames = np.repeat(p[None], 6, axis=0)
    frames[1, :, 0] *= np.float32(1.125)
    frames[3] += np.float32(0.125)
    frames[4, -1] = frames[4, 0]
    frames *= np.float32(2.0**exponent)
    rows = [
        candidates.copy() if case == "flat" else positive_canonical(f, candidates) for f in frames
    ]
    capacity = max(map(len, rows)) + 5
    cells = np.full((6, capacity, 4), -1, np.int32)
    proofs = np.full((6, capacity), -117, np.int32)
    counts = np.asarray([len(t) for t in rows], np.int32)
    counts[2] = 0
    for f, t in enumerate(rows):
        cells[f, : len(t)] = t
        proofs[f, : len(t)] = np.arange(len(t)) % 2 if f in [1, 4] else int(f == 3)
    errors = np.asarray([0, 16, 128, 64, 1, 0], np.int32)
    with wp.ScopedDevice("cuda:0"):
        points = wp.array(frames[0], dtype=wp.vec3)
        tets = wp.empty((capacity, 4), dtype=int)
        count = wp.zeros(1, dtype=int)
        proved = wp.empty(capacity, dtype=int)
        phase = wp.zeros(1, dtype=int)
        initial = wp.zeros(1, dtype=int)
        error = wp.zeros(1, dtype=int)
        statistics = wp.zeros(1, dtype=int)
        proof_statistics = wp.zeros(2, dtype=int)
        source = wp.array(frames, dtype=wp.vec3)
        source_cells = wp.array(cells, dtype=wp.vec4i)
        source_counts = wp.array(counts, dtype=int)
        source_proofs = wp.array(proofs, dtype=int)
        source_errors = wp.array(errors, dtype=int)
        weights = wp.zeros(len(p), dtype=float)
        owners, retained = _canonical.canonicalize(points, weights)
        table, buckets, bits = retained
        hierarchy = SiteHierarchy(points)
        hierarchy.tree.owners = owners
        original = [
            points,
            tets,
            count,
            hierarchy.tree,
            error,
            statistics,
            proved,
            proof_statistics,
        ]
        prepared = PreparedUnprovedQueries(*original, segment=hierarchy._segment, block_dim=block)
        output = wp.empty((6, 4, 4), dtype=int)
        wp.load_module(module=__name__, device="cuda:0", block_dim=block)
        wp.load_module(module="warpower._proved_completion", device="cuda:0", block_dim=block)
        with wp.ScopedCapture() as capture:
            phase.zero_()
            for _ in range(6):
                wp.launch(
                    produce,
                    max(len(p), capacity),
                    inputs=[
                        source,
                        source_cells,
                        source_counts,
                        source_proofs,
                        source_errors,
                        phase,
                        points,
                        tets,
                        count,
                        proved,
                        initial,
                    ],
                    block_dim=block,
                )
                table.fill_(-1)
                wp.launch(_canonical.insert, len(p), inputs=[points, weights, bits, table, buckets])
                wp.launch(_canonical.resolve, len(p), inputs=[table, buckets, owners])
                hierarchy.rebuild()
                for mode in range(4):
                    wp.launch(
                        reset,
                        1,
                        inputs=[initial, error, statistics, proof_statistics],
                        block_dim=block,
                    )
                    if mode in [0, 2]:
                        checker = check_unproved_spheres if mode == 0 else reference_without_box
                        wp.launch(checker, 32 * capacity, inputs=original, block_dim=block)
                    else:
                        wp.launch(
                            poison,
                            capacity,
                            inputs=[prepared.shapes, prepared.boxes],
                            block_dim=block,
                        )
                        prepared.pending.fill_(wp.uint32(0xFFFFFFFF))
                        prepared.unresolved.fill_(-233)
                        if mode == 1:
                            prepared.run()
                        else:
                            wp.launch(
                                prepare_unproved,
                                capacity,
                                inputs=prepared.prepare_inputs,
                                block_dim=block,
                            )
                            wp.launch(
                                invalidate, capacity, inputs=[prepared.boxes], block_dim=block
                            )
                            prepared.query()
                    wp.launch(
                        consume,
                        1,
                        inputs=[phase, mode, error, statistics, proof_statistics, output],
                        block_dim=block,
                    )
                wp.launch(advance, 1, inputs=[phase], block_dim=block)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            actual = output.numpy()
            np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
            np.testing.assert_array_equal(actual[:, 2], actual[:, 3])
            np.testing.assert_array_equal(actual[:, :, 0], np.repeat(actual[:, :1, 0], 4, axis=1))
            for f in range(6):
                n = int(counts[f])
                known = int(np.count_nonzero(proofs[f, :n]))
                np.testing.assert_array_equal(actual[f, :, 2], 11 + known)
                np.testing.assert_array_equal(actual[f, :, 3], 13 + n - known)
                if len(p) <= 8:
                    want = rational_error(frames[f], rows[f][:n], proofs[f, :n], errors[f])
                    np.testing.assert_array_equal(actual[f, :, 0], want)
            assert actual[2, 0].tolist() == [128, 7, 11, 13]
            assert actual[3, 0, 0] == 64 and actual[3, 0, 1] == 7
            np.testing.assert_array_equal(actual[0], actual[5])
