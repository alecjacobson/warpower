"""Saved proof provenance versus exhaustive and rational current-input queries."""
from itertools import combinations, product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from temporal_seed_queries import TemporalSeeds
from tests.test_grouped_query_masks import positive_canonical, rational_mask
from warpower import _canonical
from warpower._facet_seeds import CertifiedSeeds
from warpower._site_bvh import SiteHierarchy


@wp.kernel
def produce(frames: wp.array2d[wp.vec3], sources: wp.array3d[int], counts: wp.array[int],
            phase: wp.array[int], points: wp.array[wp.vec3], source: wp.array2d[int],
            count: wp.array[int], enabled: wp.array[int], available: wp.array[int]):
    i = wp.tid()
    f = phase[0]
    if i < points.shape[0]:
        points[i] = frames[f, i]
    if i < source.shape[0]:
        for k in range(4):
            source[i, k] = sources[f, i, k]
    if i == 0:
        count[0] = counts[f]
        available[0] *= enabled[f]


@wp.kernel
def consume(phase: wp.array[int], which: int, unique: wp.array2d[int], count: wp.array[int],
            live: wp.array[int], error: wp.array[int], output_count: wp.array[int],
            source_proved: wp.array[int], proved: wp.array[int], changed: wp.array[int],
            changed_count: wp.array[int], history: wp.array3d[wp.vec4i], masks: wp.array3d[int],
            meta: wp.array3d[int], source_flags: wp.array2d[int], unique_flags: wp.array2d[int],
            query_ids: wp.array2d[int]):
    i = wp.tid()
    f = phase[0]
    ids = wp.vec4i(-1)
    keep = int(0)
    if i < count[0]:
        ids = wp.vec4i(unique[i, 0], unique[i, 1], unique[i, 2], unique[i, 3])
        keep = live[i]
    history[f, which, i] = ids
    masks[f, which, i] = keep
    if i == 0:
        meta[f, which, 0] = count[0]
        meta[f, which, 1] = error[0]
        meta[f, which, 2] = output_count[0]
        meta[f, which, 3] = changed_count[0]
    if which == 1:
        source_flags[f, i] = source_proved[i]
        unique_flags[f, i] = proved[i]
        if i < query_ids.shape[1]:
            query_ids[f, i] = -1
            if i < changed_count[0]:
                query_ids[f, i] = changed[i]


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


@pytest.mark.parametrize('case', ['cube', 'octahedron', 'random33', 'random65'])
@pytest.mark.parametrize('exponent', [-60, 0, 60])
@pytest.mark.parametrize('block', [32, 256])
def test_original_source_identity_owner_promotion_and_changed_queries(case, exponent, block):
    rng = np.random.default_rng(130178)
    if case == 'cube':
        base = np.array(list(product([-1, 1], repeat=3)), np.float32)
    elif case == 'octahedron':
        base = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
    else:
        base = rng.normal(size=(int(case[6:]), 3)).astype(np.float32)
    p = np.vstack([base, base[:3]]) * np.float32(2.0**exponent)
    candidates = (Delaunay(base).simplices if case.startswith('random')
                  else np.array(list(combinations(range(len(base)), 4)), np.int32))
    cells = positive_canonical(p, candidates)
    if not case.startswith('random'):
        cells = cells[rational_mask(p, cells).astype(bool)]
    assert len(cells) > 1
    capacity = max(len(cells) + 7, len(p))
    old_source = np.full((capacity, 4), -1, np.int32)
    old_source[:len(cells)] = cells
    frames = np.repeat(p[None], 15, axis=0)
    frames[1, 5] += np.array([0.013, -0.002, 0.01], np.float32) * np.float32(2.0**exponent)
    frames[2] += rng.normal(size=p.shape).astype(np.float32) * np.float32(2.0**(exponent - 12))
    frames[3, 0, 0] = np.nextafter(p[0, 0], np.float32(-np.inf))
    frames[4, len(base) - 1] = p[0]
    if case == 'random65':
        movable = [q for q in range(3, len(base)) if q not in cells[0]]
        center = p[cells[0]].astype(np.float64).mean(axis=0).astype(np.float32)
        for frame, amount in zip([12, 13, 14], [31, 32, 33]):
            next_point = center.copy()
            for q in movable[:amount]:
                next_point[0] = np.nextafter(next_point[0], np.float32(np.inf))
                frames[frame, q] = next_point
    sources = np.repeat(old_source[None], len(frames), axis=0)
    # Valid-looking source corruption must lose its old proof even with no motion.
    alternatives = positive_canonical(p, np.array(list(combinations(range(min(len(base), 8)), 4)), np.int32))
    old_keys = {tuple(sorted(t)) for t in cells}
    extra = next(t for t in alternatives if tuple(sorted(t)) not in old_keys)
    sources[5, 0] = extra
    sources[11, 0, 0] = -1
    counts = np.full(len(frames), len(cells), np.int32)
    counts[6] //= 2
    counts[7] = 0
    counts[10] = capacity + 1
    enabled = np.ones(len(frames), np.int32)
    enabled[9] = 0

    with wp.ScopedDevice('cuda:0'):
        points = wp.array(p, dtype=wp.vec3)
        source = wp.array(old_source, dtype=int)
        count = wp.zeros(1, dtype=int)
        old_points = wp.array(p, dtype=wp.vec3)
        old_cells = wp.array(old_source, dtype=int)
        frame_data = wp.array(frames, dtype=wp.vec3)
        source_data = wp.array(sources, dtype=int)
        count_data = wp.array(counts, dtype=int)
        enable_data = wp.array(enabled, dtype=int)
        weights = wp.zeros(len(p), dtype=float)
        owners, retained = _canonical.canonicalize(points, weights)
        table, buckets, bits = retained
        hierarchy = SiteHierarchy(points)
        hierarchy.tree.owners = owners
        reference = CertifiedSeeds(points, source, count, hierarchy.tree, block_dim=block, prepare_geometry=True)
        temporal = TemporalSeeds(points, source, count, hierarchy.tree, block_dim=block, prepare_geometry=True)
        phase = wp.zeros(1, dtype=int)
        no_error = wp.zeros(1, dtype=int)
        history = wp.empty((len(frames), 2, capacity), dtype=wp.vec4i)
        masks = wp.empty((len(frames), 2, capacity), dtype=int)
        meta = wp.empty((len(frames), 2, 4), dtype=int)
        source_flags = wp.empty((len(frames), capacity), dtype=int)
        unique_flags = wp.empty_like(source_flags)
        query_ids = wp.empty((len(frames), len(p)), dtype=int)
        wp.load_module(module=__name__, device='cuda:0', block_dim=block)

        def rebuild():
            hierarchy.rebuild()
            table.fill_(-1)
            wp.launch(_canonical.insert, len(p), inputs=[points, weights, bits, table, buckets])
            wp.launch(_canonical.resolve, len(p), inputs=[table, buckets, owners])

        with wp.ScopedCapture() as cap:
            wp.copy(points, old_points)
            wp.copy(source, old_cells)
            count.fill_(len(cells))
            rebuild()
            # The old cells were independently certified above. Snapshot their
            # exact source identity; the current source may be remapped/altered.
            temporal.saved_points.fill_(wp.vec3(wp.nan))
            temporal.saved_owners.fill_(-13)
            temporal.saved_source.fill_(-19)
            temporal.saved_count.fill_(-23)
            temporal.available.fill_(-1)
            temporal.remember(no_error)
            wp.launch(produce, max(len(p), capacity), inputs=[frame_data, source_data, count_data,
                phase, points, source, count, enable_data, temporal.available], block_dim=block)
            rebuild()
            temporal.changed_count.fill_(130)
            temporal.changed.fill_(-29)
            temporal.changed_flags.fill_(0)
            temporal.source_proved.fill_(1)
            temporal.proved.fill_(1)
            for which, query in enumerate([reference, temporal]):
                query.run()
                wp.launch(consume, capacity, inputs=[phase, which, query.unique, query.unique_count,
                    query.live, query.error, query.tetrahedra_count, temporal.source_proved,
                    temporal.proved, temporal.changed, temporal.changed_count, history, masks,
                    meta, source_flags, unique_flags, query_ids], block_dim=block)
            wp.launch(advance, 1, inputs=[phase], block_dim=block)
        for repeat in range(2):
            phase.zero_()
            for _ in frames:
                wp.capture_launch(cap.graph)
            h, m, md = history.numpy(), masks.numpy(), meta.numpy()
            sf, uf, qi = source_flags.numpy(), unique_flags.numpy(), query_ids.numpy()
            for f, current in enumerate(frames):
                np.testing.assert_array_equal(md[f, 0, :3], md[f, 1, :3])
                nc = int(md[f, 1, 0])
                dictionaries = [{tuple(sorted(t)): int(k) for t, k in zip(h[f, j, :nc], m[f, j, :nc])} for j in range(2)]
                assert dictionaries[0] == dictionaries[1]
                np.testing.assert_array_equal(m[f, :, nc:], 0)
                if f == 10:
                    assert md[f, 1, 1] == 4 and nc == 0
                else:
                    assert md[f, 1, 1] == 0
                    if len(p) <= 16:
                        np.testing.assert_array_equal(m[f, 1, :nc], rational_mask(current, h[f, 1, :nc]))
                _, first, inverse = np.unique(current, axis=0, return_index=True, return_inverse=True)
                current_owners = first[inverse]
                _, old_first, old_inverse = np.unique(p, axis=0, return_index=True, return_inverse=True)
                old_owners = old_first[old_inverse]
                expected = np.zeros(capacity, np.int32)
                if enabled[f] and 0 <= counts[f] <= capacity:
                    for i in range(min(counts[f], len(cells))):
                        ids = sources[f, i]
                        if np.array_equal(ids, old_source[i]) and np.all(current_owners[ids] == ids) and np.all(old_owners[ids] == ids) and np.array_equal(current[ids], p[ids]):
                            expected[i] = 1
                np.testing.assert_array_equal(sf[f], expected)
                required = [q for q in first if not enabled[f] or old_owners[q] != q or not np.array_equal(current[q], p[q])]
                assert md[f, 1, 3] == len(required)
                assert sorted(qi[f, :len(required)]) == sorted(required)
                np.testing.assert_array_equal(qi[f, len(required):], -1)
                for t, proved in zip(h[f, 1, :nc], uf[f, :nc]):
                    if proved:
                        assert tuple(sorted(t)) in old_keys
                        assert np.array_equal(current[t], p[t])
                        assert np.all(current_owners[t] == t) and np.all(old_owners[t] == t)
            assert md[0, 1, 3] == 0 and np.all(sf[0, :len(cells)] == 1)
            assert sf[5, 0] == 0
            assert len(base) in qi[3, :md[3, 1, 3]]  # Promoted unchanged duplicate.
            if case == 'random65':
                for f, amount in zip([12, 13, 14], [31, 32, 33]):
                    assert md[f, 1, 3] == amount and sf[f, 0] == 1
                    key = tuple(sorted(cells[0]))
                    row = {tuple(sorted(t)): int(k) for t, k in zip(h[f, 1, :md[f, 1, 0]], m[f, 1, :md[f, 1, 0]])}
                    assert row[key] == 0  # Changed sites were put strictly inside this cached sphere.
