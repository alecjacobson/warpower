from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_facet_completion import canonical, execute
from tests.test_facet_seeds import fixture as seed_fixture
from tests.test_local_certificate import positive
from tests.test_predicates import exact_reference
from warpower._delaunay_certificate import DelaunayCertificate
from warpower._proved_completion import SeededFacetCompletion, unique_proved_pool
from warpower._reuse_delaunay import ReuseDelaunayWorkspace


def completion(points, seeds, **kwargs):
    seed_filter, _, retained = seed_fixture(points, seeds)
    kwargs.setdefault("capacity", max(len(seeds), 16 * len(points)))
    return SeededFacetCompletion(seed_filter, query_block_dim=32, **kwargs), retained


def global_certificate(workspace):
    c = DelaunayCertificate(
        workspace.points,
        workspace.tetrahedra,
        workspace.tetrahedra_count,
        workspace.tree,
        point_parallel=True,
    )
    c._anchors = None
    with wp.ScopedCapture() as capture:
        c.run()
    wp.capture_launch(capture.graph)
    assert c.error.numpy()[0] == 0


def exact_symbolic_cells(points, tets):
    """Rational determinants of the lifting polynomial, without GPU predicates."""
    owners = np.unique(points, axis=0, return_index=True)[1]
    weights = []
    for i in range(len(points)):
        value = i + 1
        value = ((value ^ (value >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
        value = ((value ^ (value >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
        value ^= value >> 16
        weights.append(Fraction(value & 0xFFFFFF, 1 << 23) - 1)
    assert set(tets.ravel()) == set(owners)
    for tet in tets:
        for query in owners:
            if query in tet:
                continue
            ids = [*tet, query]
            rows = points[ids]
            orientation, sign = exact_reference(rows, np.zeros(5))
            assert orientation > 0
            if sign == 0:
                _, sign = exact_reference(rows, [weights[i] for i in ids])
            if sign == 0:
                for site in sorted(ids):
                    _, sign = exact_reference(rows, [int(i == site) for i in ids])
                    if sign != 0:
                        break
            assert sign < 0, (tet, query, sign)


@pytest.mark.parametrize("subset", ["one", "scattered", "hole", "full"])
def test_proved_fragments_match_scipy_and_ordinary_global_certificate(subset):
    points = np.random.default_rng(941).normal(size=(32, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    seeds = {"one": full[:1], "scattered": full[::7], "hole": full[1:], "full": full}[subset]
    w, retained = completion(points, seeds)
    error, actual = execute(w)
    assert error == 0
    np.testing.assert_array_equal(canonical(actual), canonical(full))
    skipped, checked = w._certificate.proof_statistics.numpy()
    assert skipped >= len(seeds)
    assert checked == 0 if subset == "full" else checked > 0
    global_certificate(w)


@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
@pytest.mark.parametrize("shape", ["cube", "octahedron"])
def test_exhaustive_candidates_with_duplicates_match_rational_symbolic_oracle(shape, scale):
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    if shape == "octahedron":
        points = np.vstack([np.eye(3), -np.eye(3)]).astype(np.float32)
    points = np.vstack([points, points[:2]]) * np.float32(scale)
    seeds = np.array(list(combinations(range(len(points)), 4)), np.int32)
    seeds = np.vstack([seeds, seeds[:, [1, 0, 2, 3]]])
    w, retained = completion(points, seeds)
    error, actual = execute(w)
    assert error == 0
    assert w.rounds.numpy()[0] == 0
    assert w._certificate.proof_statistics.numpy().tolist() == [len(actual), 0]
    global_certificate(w)
    exact_symbolic_cells(points, actual)


@pytest.mark.parametrize("first", ["old", "new"])
def test_hash_collisions_and_duplicate_winners_keep_source_provenance(first):
    points = np.random.default_rng(942).normal(size=(9, 3)).astype(np.float32)
    buckets = {}
    for key in combinations(range(len(points)), 4):
        slot = (key[0] * 73856093 ^ key[1] * 19349663 ^ key[2] * 83492791 ^ key[3] * 2654435761) & 7
        buckets.setdefault(slot, []).append(key)
    keys = np.array(next(v[:5] for v in buckets.values() if len(v) >= 5), np.int32)
    pool = np.zeros((40, 4), np.int32)
    positions, ids = [0, 7, 15, 16, 20, 25, 39], [0, 1, 2, 1, 0, 3, 4]
    pool[positions] = keys[ids]
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    source = wp.array(pool, dtype=wp.vec4i, device=p.device)
    table = wp.full(8, -1, dtype=int, device=p.device)
    output = wp.empty((5, 4), dtype=int, device=p.device)
    count, error = wp.zeros(1, dtype=int, device=p.device), wp.zeros(1, dtype=int, device=p.device)
    proved = wp.full(5, -1, dtype=int, device=p.device)
    for phase in [first, "new" if first == "old" else "old"]:
        live = np.zeros(len(pool), np.int32)
        live[[i for i in positions if (i < 16) == (phase == "old")]] = 1
        wp.launch(
            unique_proved_pool,
            len(pool),
            inputs=[
                p,
                source,
                wp.array(live, device=p.device),
                table,
                output,
                count,
                error,
                16,
                proved,
            ],
            device=p.device,
        )
    assert error.numpy()[0] == 0 and count.numpy()[0] == 5
    actual = {tuple(sorted(t)): int(bit) for t, bit in zip(output.numpy(), proved.numpy())}
    expected = {tuple(k): int(i < 3 if first == "old" else i == 2) for i, k in enumerate(keys)}
    assert actual == expected


@wp.kernel
def corrupt_new(tets: wp.array2d[int], count: wp.array[int], proved: wp.array[int], bad: wp.vec4i):
    i = wp.tid()
    if i < count[0] and proved[i] == 0:
        for k in range(4):
            tets[i, k] = bad[k]


def test_valid_retained_proofs_do_not_hide_nonempty_new_cells():
    points = np.random.default_rng(943).normal(size=(24, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    valid_keys = {tuple(sorted(t)) for t in full}
    bad_key = next(k for k in combinations(range(len(points)), 4) if k not in valid_keys)
    bad = positive(points, np.array([bad_key], np.int32))[0]
    assert any(
        exact_reference(points[[*bad, q]], np.zeros(5))[1] > 0
        for q in range(len(points))
        if q not in bad
    )
    w, retained = completion(points, full[:1])
    original = w._deduplicate

    def inject():
        original()
        wp.launch(
            corrupt_new,
            w.capacity,
            inputs=[w._work, w._count, w._certificate._input_proved, wp.vec4i(*map(int, bad))],
            device=w.device,
        )

    w._deduplicate = inject
    wp.load_module(module=__name__, device=w.device)
    error, cells = execute(w)
    assert error != 0 and len(cells) == 0
    assert w.certificate_error.numpy()[0] & 8
    assert np.all(w._certificate.proof_statistics.numpy() > 0)


@pytest.mark.parametrize("limit", ["rounds", "capacity"])
def test_proved_fragment_exhaustion_never_publishes_a_partial_hull(limit):
    points = np.random.default_rng(944).normal(size=(24, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    options = dict(max_rounds=0) if limit == "rounds" else dict(capacity=1)
    w, retained = completion(points, full[:1], **options)
    error, cells = execute(w)
    assert error == (16 if limit == "rounds" else 4)
    assert len(cells) == 0


def test_invalid_seed_counts_and_rejected_seeds_cannot_reuse_old_proofs():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]], np.float32)
    full = positive(points, Delaunay(points).simplices)
    bad = np.array([[-1, 0, 1, 2], [0, 1, 2, 99], [0, 0, 1, 2], [0, 1, 2, 3]], np.int32)
    seeds = np.vstack([bad, full])
    w, retained = completion(points, seeds)
    with wp.ScopedCapture() as capture:
        w.run()
    for n in [len(seeds), -1, len(seeds) + 1, 0, 4, len(seeds)]:
        w._seed_filter.count.fill_(n)
        wp.capture_launch(capture.graph)
        if n == len(seeds):
            assert w.error.numpy()[0] == 0
            actual = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
            np.testing.assert_array_equal(canonical(actual), canonical(full))
            global_certificate(w)
        else:
            assert w.error.numpy()[0] != 0 and w.tetrahedra_count.numpy()[0] == 0
            assert w._certificate.proof_statistics.numpy().tolist() == [0, 0]


def test_promoted_duplicate_and_restored_owner_receive_fresh_symbolic_proofs():
    points = np.array(list(product([-1, 1], repeat=3)), np.float32)
    points = np.vstack([points, points[:1]])
    moved = points.copy()
    moved[0, 0] = np.nextafter(moved[0, 0], np.float32(-np.inf))
    w = ReuseDelaunayWorkspace(points)
    with wp.ScopedCapture() as capture:
        w.run()
    for frame in [points, points, moved, points]:
        w.points.assign(frame)
        wp.capture_launch(capture.graph)
        assert w.error.numpy()[0] == 0
        tets = w.tetrahedra.numpy()[: w.tetrahedra_count.numpy()[0]]
        exact_symbolic_cells(frame, tets)
