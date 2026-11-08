"""Captured query reuse across deformation, cutoff changes, reorder, invalid input and recovery."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_domain import cube
from tests.test_extraction import orient
from warpower import SurfaceDomain, _extraction
from warpower import _surface_query_cache as k
from warpower._selective import classify_retained
from warpower._surface_bvh import SurfaceTree, closest_point, material_side
from warpower._surface_query_cache import SurfaceQueryCache as Cache

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def produce(
    phase: wp.array[int],
    surfaces: wp.array2d[wp.vec3],
    queries: wp.array2d[wp.vec3d],
    all_limits: wp.array2d[wp.float64],
    all_counts: wp.array[int],
    all_refit: wp.array[int],
    surface: wp.array[wp.vec3],
    centers: wp.array[wp.vec3d],
    limits: wp.array[wp.float64],
    count: wp.array[int],
    refit: wp.array[int],
    stats: wp.array[int],
):
    i = wp.tid()
    p = phase[0]
    if i < surface.shape[0]:
        surface[i] = surfaces[p, i]
    if i < centers.shape[0]:
        centers[i] = queries[p, i]
        limits[i] = all_limits[p, i]
    if i == 0:
        count[0] = all_counts[p]
        refit[0] = all_refit[p]
    if i < stats.shape[0]:
        stats[i] = 0


@wp.kernel
def query(
    tree: SurfaceTree,
    centers: wp.array[wp.vec3d],
    limits: wp.array[wp.float64],
    count: wp.array[int],
    error: wp.array[int],
    current: k.QueryState,
    previous: k.QueryState,
    table: wp.array[int],
    stats: wp.array[int],
):
    i = wp.tid()
    current.valid[i] = 0
    current.rays[i] = 0
    current.nearest[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    center = centers[i]
    limit = limits[i]
    j = k.lookup(center, previous, table)
    if j >= 0:
        if previous.rays[j] != 0:
            wp.atomic_add(stats, 0, 1)
        if previous.nearest[j] != 0 and wp.cast(limit, wp.uint64) == wp.cast(
            previous.limits[j], wp.uint64
        ):
            wp.atomic_add(stats, 1, 1)
    side = material_side(tree, center)
    cached = k.cached_side(tree, center, previous, table)
    hit = closest_point(tree, center, limit)
    other = k.cached_nearest(tree, center, limit, previous, table)
    equal = (
        side == cached
        and hit.face == other.face
        and wp.cast(hit.distance2, wp.uint64) == wp.cast(other.distance2, wp.uint64)
        and k.same_center(hit.point, other.point)
    )
    if not equal:
        wp.atomic_add(stats, 2, 1)
    current.valid[i] = 1
    current.centers[i] = center
    current.rays[i] = 1
    current.sides[i] = cached
    current.nearest[i] = 1
    current.limits[i] = limit
    current.faces[i] = other.face
    current.distances[i] = other.distance2
    current.hit_points[i] = other.point


@wp.kernel
def consume(
    phase: wp.array[int],
    condition: wp.array[int],
    error: wp.array[int],
    count: wp.array[int],
    stats: wp.array[int],
    rows: wp.array2d[int],
):
    p = phase[0]
    rows[p, 0] = error[0]
    rows[p, 1] = count[0]
    for j in range(3):
        rows[p, j + 2] = stats[j]
    phase[0] = p + 1
    condition[0] = int(p + 1 < rows.shape[0])


def test_captured_queries_refit_cutoff_failure_recovery_and_saturation():
    surface, faces = cube()
    domain = SurfaceDomain(surface, faces)
    n = 64
    phases = 13
    points = np.random.default_rng(218).uniform(-3, 3, (n, 3)).astype(np.float64)
    points[:7] = [[1.5, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0], [-0.0, 0, 0], [3, 3, 3]]
    surfaces = np.repeat(surface[None], phases, axis=0)
    surfaces[2:5] *= 2
    surfaces[4, 0] = np.nan
    queries = np.stack([np.roll(points, i % 3, axis=0) for i in range(phases)])
    limits = np.full((phases, n), 1e150, np.float64)
    limits[8] = 0.1
    counts = np.full(phases, n, np.int32)
    counts[6] = 0
    counts[11] = n + 1
    refits = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], np.int32)
    cache = Cache(domain, n)
    surface_data = wp.array(surfaces, dtype=wp.vec3)
    query_data = wp.array(queries, dtype=wp.vec3d)
    limit_data = wp.array(limits, dtype=wp.float64)
    count_data = wp.array(counts, dtype=int)
    refit_data = wp.array(refits, dtype=int)
    current_points = wp.empty(n, dtype=wp.vec3d)
    current_limits = wp.empty(n, dtype=wp.float64)
    phase = wp.zeros(1, dtype=int)
    condition = wp.ones(1, dtype=int)
    count = wp.zeros(1, dtype=int)
    refit = wp.zeros(1, dtype=int)
    error = wp.zeros(1, dtype=int)
    source_error = wp.zeros(1, dtype=int)
    stats = wp.zeros(3, dtype=int)
    rows = wp.zeros((phases, 5), dtype=int)
    wp.load_module(module=__name__, device=domain.device)

    def step():
        wp.launch(
            produce,
            max(n, len(surface)),
            inputs=[
                phase,
                surface_data,
                query_data,
                limit_data,
                count_data,
                refit_data,
                domain.points,
                current_points,
                current_limits,
                count,
                refit,
                stats,
            ],
        )
        wp.capture_if(refit, domain.refit)
        wp.launch(_extraction.input_ready, 1, inputs=[domain.error, source_error, count, n, error])
        wp.launch(
            query,
            n,
            inputs=[
                domain.tree,
                current_points,
                current_limits,
                count,
                error,
                cache.current,
                cache.previous,
                cache.table,
                stats,
            ],
            block_dim=32,
        )
        cache.after(error)
        wp.launch(consume, 1, inputs=[phase, condition, error, count, stats, rows])

    with wp.ScopedCapture() as capture:
        phase.zero_()
        condition.fill_(1)
        wp.capture_while(condition, step)
    results = []
    for _ in range(2):
        wp.capture_launch(capture.graph)
        r = rows.numpy()
        results.append(r.tolist())
        np.testing.assert_array_equal(r[:, 4], 0)
        assert r[4, 0] != 0 and r[11, 0] == 2
        np.testing.assert_array_equal(np.delete(r[:, 0], [4, 11]), 0)
        for index in [0, 2, 4, 5, 6, 7, 11, 12]:
            np.testing.assert_array_equal(r[index, 2:4], 0)
        for index in [1, 3, 10]:
            np.testing.assert_array_equal(r[index, 2:4], [n, n])
        for index in [8, 9]:
            np.testing.assert_array_equal(r[index, 2:4], [n, 0])
    np.testing.assert_array_equal(results[0], results[1])
    domain._query_epoch.fill_(9223372036854775806)
    wp.capture_launch(capture.graph)
    saturated = rows.numpy()
    np.testing.assert_array_equal(saturated[:, 2:5], 0)
    np.testing.assert_array_equal(saturated[:, 0], np.array(results[0])[:, 0])
    assert domain._query_epoch.numpy()[0] == 9223372036854775807


@pytest.mark.parametrize("mode", ["full", "selection", "retained"])
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_cold_warm_classification_matches_ordinary(mode, scale):
    surface, faces = cube()
    domain = SurfaceDomain((surface * scale).astype(np.float32), faces)
    source = np.random.default_rng(218).uniform(-3, 3, (32, 3)).astype(np.float32)
    cells = orient(source, Delaunay(source).simplices)
    n = len(cells)
    capacity = n + 9
    padded = np.full((capacity, 4), -7, np.int32)
    padded[:n] = cells
    points = wp.array((source * scale).astype(np.float32), dtype=wp.vec3)
    tets = wp.array(padded, dtype=int)
    count = wp.full(1, n, dtype=int)
    cache = Cache(domain, capacity)
    reference = {
        "full": _extraction.classify,
        "selection": _extraction.classify_selection,
        "retained": classify_retained,
    }[mode]
    target = {
        "full": k.classify_cached,
        "selection": k.selection_cached,
        "retained": k.retained_cached,
    }[mode]
    phases = []
    for _ in range(3):
        error = wp.zeros(1, dtype=int)
        keep = wp.full(capacity, 77, dtype=int)
        quality = wp.full(capacity, 77, dtype=wp.float64)
        arrays = [error, keep, quality]
        args = [domain.tree, points, tets, count, 0.4, error]
        if mode == "retained":
            args += [keep, quality]
        else:
            center = wp.full(capacity, wp.vec3d(77), dtype=wp.vec3d)
            radius = wp.full(capacity, 77, dtype=wp.float64)
            distance = wp.full(capacity, 77, dtype=wp.float64)
            volume = wp.full(capacity, 77, dtype=wp.float64)
            arrays += [center, radius, distance, volume]
            args += [center, radius, distance, keep, volume, quality]
        phases.append((args, arrays))
    with wp.ScopedCapture() as capture:
        wp.launch(reference, capacity, inputs=phases[0][0], block_dim=32)
        cache.launch(target, capacity, phases[1][0])
        cache.launch(target, capacity, phases[2][0])
    wp.capture_launch(capture.graph)
    results = [[a.numpy() for a in arrays] for _, arrays in phases]
    assert results[0][0][0] == 0
    for other in results[1:]:
        np.testing.assert_array_equal(other[0], results[0][0])
        np.testing.assert_array_equal(other[1], results[0][1])
        for actual, expected in zip(other[2:], results[0][2:]):
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    for result in results:
        np.testing.assert_array_equal(result[1][n:], 0)
        for values in result[2:]:
            np.testing.assert_array_equal(values[n:], 77)
