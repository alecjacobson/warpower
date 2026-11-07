"""Captured input changes compare prepared query filter with exhaustive masks."""

from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_predicates import determinant, exact_reference
from warpower import _canonical
from warpower import _prepared_seeds as ps
from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, outside, symbolic_sphere
from warpower._facet_seeds import reset_live
from float32_seed_queries import Float32GroupedQueries as GroupedSeedQueries
from warpower._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def reference_check(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    # Match _facet_seeds.check_seeds, including its exact/symbolic fallback.
    # Keeping the original kernel separate preserves ordinary construction.
    for q in range(lane, points.shape[0], 32):
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p) or outside(shape, p, p):
            continue
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


@wp.kernel
def produce(
    frames: wp.array2d[wp.vec3],
    cells: wp.array2d[wp.vec4i],
    counts: wp.array[int],
    phase: wp.array[int],
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
):
    i = wp.tid()
    f = phase[0]
    if i < points.shape[0]:
        points[i] = frames[f, i]
    if i < tets.shape[0]:
        ids = cells[f, i]
        for k in range(4):
            tets[i, k] = ids[k]
    if i == 0:
        count[0] = counts[f]


@wp.kernel
def consume(
    phase: wp.array[int],
    live: wp.array[int],
    error: wp.array[int],
    which: int,
    masks: wp.array3d[int],
    failures: wp.array2d[int],
):
    i = wp.tid()
    masks[phase[0], which, i] = live[i]
    if i == 0:
        failures[phase[0], which] = error[0]


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def positive_canonical(points, candidates):
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    owners = first[inverse]
    result = []
    for t in candidates:
        ids = owners[t].copy()
        if len(set(ids)) != 4:
            continue
        sign = -determinant([[*map(float, points[i]), 1] for i in ids])
        if sign:
            result.append(ids if sign > 0 else ids[[1, 0, 2, 3]])
    return np.asarray(result, np.int32).reshape(-1, 4)


def rational_mask(points, cells):
    owners = np.unique(points, axis=0, return_index=True)[1]
    weights = []
    for i in range(len(points)):
        v = i + 1
        v = ((v ^ (v >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        v = ((v ^ (v >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        v ^= v >> 16
        weights.append(Fraction(v & 0xFFFFFF, 1 << 23) - 1)
    result = []
    for t in cells:
        keep = 1
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
            assert value != 0
            if value > 0:
                keep = 0
                break
        result.append(keep)
    return np.asarray(result, np.int32)


@pytest.mark.parametrize(
    "case", ["cube", "octahedron", "thin", "random33", "random257", "torus4096"]
)
@pytest.mark.parametrize("exponent", [-60, 0, 60])
@pytest.mark.parametrize("block", [32, 256])
def test_captured_producers_preserve_exhaustive_and_rational_masks(case, exponent, block):
    rng = np.random.default_rng(170169)
    if case == "torus4096":
        saved = np.load(
            "results/validation/vtm/analytic-filters/analytic-final/torus4096-after.npz"
        )
        p, candidates = saved["points"], saved["full_tetrahedra"]
        candidates = candidates[np.linspace(0, len(candidates) - 1, 32, dtype=int)]
    else:
        p = np.array(list(product([-1, 1], repeat=3)), np.float32)
        if case == "octahedron":
            p = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
        if case.startswith("random"):
            p = rng.normal(size=(int(case[6:]), 3)).astype(np.float32)
            candidates = Delaunay(p).simplices
            candidates = candidates[np.linspace(0, len(candidates) - 1, 32, dtype=int)]
        else:
            candidates = np.asarray(list(combinations(range(len(p)), 4)), np.int32)
        if case == "thin":
            p[:, 2] *= np.float32(2**-30)
    p = np.vstack([p, p[:3]]) * np.float32(2.0**exponent)
    moved = p * np.float32(2) + np.array([4, 7, -5], np.float32) * np.float32(2.0**exponent)
    moved[5] += np.array([0.125, -0.25, 0.5], np.float32) * np.float32(2.0**exponent)
    permuted = np.roll(p, 3, axis=0)
    unrelated = rng.normal(size=p.shape).astype(np.float32) * np.float32(2.0**exponent)
    unrelated[-3:] = unrelated[:3]
    frames = np.stack([p, moved, permuted, unrelated, unrelated, p])
    rows = [positive_canonical(f, candidates) for f in frames]
    counts = np.array([len(t) for t in rows], np.int32)
    counts[2] //= 2
    counts[4] = 0
    capacity = max(map(len, rows)) + 7
    cells = np.full((6, capacity, 4), -1, np.int32)
    for i, t in enumerate(rows):
        cells[i, : len(t)] = t
    device = "cuda:0"
    source = wp.array(frames, dtype=wp.vec3, device=device)
    source_cells = wp.array(cells, dtype=wp.vec4i, device=device)
    source_counts = wp.array(counts, dtype=int, device=device)
    points = wp.array(p, dtype=wp.vec3, device=device)
    tets = wp.empty((capacity, 4), dtype=int, device=device)
    count = wp.zeros(1, dtype=int, device=device)
    phase = wp.zeros(1, dtype=int, device=device)
    weights = wp.zeros(len(p), dtype=float, device=device)
    owners, retained = _canonical.canonicalize(points, weights)
    table, buckets, bits = retained
    hierarchy = SiteHierarchy(points)
    hierarchy.tree.owners = owners
    live = wp.empty(capacity, dtype=int, device=device)
    errors = wp.zeros(1, dtype=int, device=device)
    prepared = GroupedSeedQueries(
        points, tets, count, hierarchy.tree, live, errors, block_dim=block
    )
    kernels = [reference_check, ps.check, None]
    masks = wp.empty((6, len(kernels), capacity), dtype=int, device=device)
    failures = wp.empty((6, len(kernels)), dtype=int, device=device)
    for module in [__name__, "warpower._facet_seeds"]:
        wp.load_module(module=module, device=device, block_dim=block)
    with wp.ScopedCapture() as cap:
        phase.zero_()
        for _ in range(6):
            wp.launch(
                produce,
                max(len(p), capacity),
                inputs=[source, source_cells, source_counts, phase, points, tets, count],
                device=device,
                block_dim=block,
            )
            table.fill_(-1)
            wp.launch(
                _canonical.insert,
                len(p),
                inputs=[points, weights, bits, table, buckets],
                device=device,
            )
            wp.launch(_canonical.resolve, len(p), inputs=[table, buckets, owners], device=device)
            hierarchy.rebuild()
            errors.zero_()
            wp.launch(
                ps.prepare,
                capacity,
                inputs=prepared._prepare_inputs,
                device=device,
                block_dim=block,
            )
            for j, kernel in enumerate(kernels):
                live.fill_(-170)
                errors.zero_()
                wp.launch(
                    reset_live, capacity, inputs=[count, live], device=device, block_dim=block
                )
                if kernel is None:
                    prepared.lower.fill_(wp.vec3(wp.nan))
                    prepared.upper.fill_(wp.vec3(wp.nan))
                    prepared.run()
                else:
                    wp.launch(
                        kernel,
                        32 * capacity,
                        inputs=prepared._check_inputs,
                        device=device,
                        block_dim=block,
                    )
                wp.launch(
                    consume,
                    capacity,
                    inputs=[phase, live, errors, j, masks, failures],
                    device=device,
                    block_dim=block,
                )
            wp.launch(advance, 1, inputs=[phase], device=device, block_dim=block)
    expected = []
    if len(p) <= 16:
        for i in range(6):
            m = np.zeros(capacity, np.int32)
            m[: counts[i]] = rational_mask(frames[i], rows[i][: counts[i]])
            expected.append(m)
    for _ in range(2):
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(failures.numpy(), 0)
        actual = masks.numpy()
        for j in range(1, len(kernels)):
            np.testing.assert_array_equal(actual[:, j], actual[:, 0])
        for i in range(6):
            np.testing.assert_array_equal(actual[i, :, counts[i] :], 0)
            if expected:
                np.testing.assert_array_equal(actual[i, 0], expected[i])
