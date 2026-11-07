from itertools import product

import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay

from tests.test_facet_completion import canonical, execute
from tests.test_facet_seeds import fixture as seed_fixture
from tests.test_local_certificate import positive
from tests.test_predicates import determinant
from tests.test_proved_completion import completion, global_certificate
from warpower._delaunay_certificate import check_faces
from warpower._facet_completion import search_faces
from warpower._proved_completion import record_face_support, search_unproved_faces


def oriented_face(tets, index):
    tet, opposite = divmod(index, 4)
    ids = np.delete(tets[tet], opposite)
    if opposite % 2:
        ids[[0, 1]] = ids[[1, 0]]
    return ids


def support_oracle(points, tets, faces):
    owners = np.unique(points, axis=0, return_index=True)[1]
    rows = [[float(v) for v in p] + [1] for p in points]
    return np.array(
        [
            all(
                -determinant([rows[i] for i in [*oriented_face(tets, face), q]]) <= 0
                for q in owners
            )
            for face in faces
        ],
        np.int32,
    )


@pytest.mark.parametrize("depth", [0, 5])
@pytest.mark.parametrize("scale", [2.0**-60, 1.0, 2.0**60])
def test_support_matches_rational_oracle_and_preserves_candidates_with_strided_tails(depth, scale):
    points = np.vstack(
        [list(product([-1, 1], repeat=3)), np.random.default_rng(138).uniform(-0.7, 0.7, (7, 3))]
    ).astype(np.float32)
    candidates = Delaunay(points).simplices
    # Qhull may include flat cells on the cube's coplanar hull facets. The
    # direct-kernel fixture needs nonflat cells, not a symbolic triangulation.
    signs = [-determinant([[*map(float, points[i]), 1] for i in t]) for t in candidates]
    tets = np.array(
        [t if sign > 0 else t[[1, 0, 2, 3]] for t, sign in zip(candidates, signs) if sign], np.int32
    )
    points = np.vstack([points, points[:3]]) * np.float32(scale)
    seed, _, retained = seed_fixture(points, tets)
    # Deliberately include interior faces, repeat faces, shuffle their slots,
    # and provide more slots than support workers. Depth5 exercises early leaves.
    faces = np.concatenate([np.arange(4 * len(tets)), [3, 0, 1, 3]])
    np.random.default_rng(139).shuffle(faces)
    expected = support_oracle(points, tets, faces)
    assert expected.any() and not expected.all()
    f = wp.array(faces.astype(np.int32), device=seed.points.device)
    count = wp.full(1, len(faces), dtype=int, device=seed.points.device)
    supported = wp.full(len(faces), 1, dtype=int, device=seed.points.device)
    errors = wp.zeros(2, dtype=int, device=seed.points.device)
    workers = 3
    for i, kernel in enumerate([check_faces, record_face_support]):
        args = [seed.points, seed.source, f, count, seed.tree, errors[i : i + 1], depth, workers]
        if i:
            args += [supported]
        wp.launch(
            kernel, workers * (1 << depth), inputs=args, block_dim=32, device=seed.points.device
        )
    np.testing.assert_array_equal(supported.numpy(), expected)
    np.testing.assert_array_equal(errors.numpy(), [64, 64])
    # Padded candidate rows must be initialized too, even when every live face
    # is proved. Existing candidates cannot leak across graph invocations.
    search_workers = 3
    size = (len(faces) + 7) * search_workers
    outputs = [wp.full(size, 12345, dtype=int, device=seed.points.device) for _ in range(2)]
    statistics = wp.zeros(2, dtype=int, device=seed.points.device)
    errors.zero_()
    for i, kernel in enumerate([search_faces, search_unproved_faces]):
        args = [
            seed.points,
            seed.tree.owners,
            seed.source,
            f,
            count,
            outputs[i],
            search_workers,
            errors[i : i + 1],
        ]
        if i:
            args += [supported, statistics]
        wp.launch(kernel, size, inputs=args, block_dim=32, device=seed.points.device)
    np.testing.assert_array_equal(errors.numpy(), [0, 0])
    np.testing.assert_array_equal(outputs[0].numpy(), outputs[1].numpy())
    np.testing.assert_array_equal(
        outputs[1].numpy().reshape(-1, search_workers)[: len(faces)][expected != 0], -1
    )
    np.testing.assert_array_equal(outputs[1].numpy()[len(faces) * search_workers :], -1)
    np.testing.assert_array_equal(
        statistics.numpy(), [expected.sum(), len(expected) - expected.sum()]
    )


@pytest.mark.parametrize("code", [2, 16])
def test_geometric_failure_prevents_consuming_apparently_supported_faces(code):
    points = np.random.default_rng(140).normal(size=(24, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    w, retained = completion(points, full[:1])
    check = w._certificate._hull_support

    def fail(tree):
        check(tree)
        # Simulate an incomplete support traversal. Its default ones must never
        # become trusted when any worker reports a geometric error.
        w._certificate._face_supported.fill_(1)
        w._certificate.error.fill_(code)

    w._certificate._hull_support = fail
    w._candidates.fill_(12345)
    error, cells = execute(w)
    assert error != 0 and len(cells) == 0
    assert w.rounds.numpy()[0] == 0
    np.testing.assert_array_equal(w._certificate.face_statistics.numpy(), [0, 0])
    np.testing.assert_array_equal(w._candidates.numpy(), 12345)
    w._certificate._hull_support = check
    error, cells = execute(w)
    assert error == 0
    np.testing.assert_array_equal(canonical(cells), canonical(full))
    global_certificate(w)


def test_each_completion_round_checks_its_newly_exposed_boundary_and_resets_support():
    points = np.random.default_rng(141).normal(size=(48, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    w, retained = completion(points, full[:1])
    c = w._certificate
    # Force a multi-round fragment; remember each support result on device so
    # the independent host oracle can validate every boundary ordering afterward.
    slots = 64
    counts = wp.zeros(slots, dtype=int, device=w.device)
    faces = wp.empty((slots, len(c._boundary)), dtype=int, device=w.device)
    supported = wp.empty_like(faces)
    tets = wp.empty((slots, w.capacity, 4), dtype=int, device=w.device)
    check = c._hull_support

    def record(tree):
        check(tree)
        wp.launch(
            snapshot,
            len(c._boundary),
            inputs=[
                w.rounds,
                c._boundary_count,
                c._boundary,
                c._face_supported,
                c.tetrahedra,
                counts,
                faces,
                supported,
                tets,
            ],
            device=w.device,
        )

    c._hull_support = record
    wp.load_module(module=__name__, device=w.device)
    error, actual = execute(w)
    assert error == 0 and w.rounds.numpy()[0] > 1
    np.testing.assert_array_equal(canonical(actual), canonical(full))
    saved_counts, saved_faces, saved_support, saved_tets = (
        counts.numpy(),
        faces.numpy(),
        supported.numpy(),
        tets.numpy(),
    )
    for r in range(int(w.rounds.numpy()[0]) + 1):
        n = saved_counts[r]
        expected = support_oracle(points, saved_tets[r], saved_faces[r, :n])
        np.testing.assert_array_equal(saved_support[r, :n], expected)
    assert np.all(c.face_statistics.numpy() > 0)
    # A second invocation starts from the same one-cell seed and must perform
    # the same work even if every stale support bit was set to the wrong value.
    previous = c.face_statistics.numpy()
    c._face_supported.zero_()
    error, again = execute(w)
    assert error == 0
    np.testing.assert_array_equal(canonical(again), canonical(full))
    np.testing.assert_array_equal(c.face_statistics.numpy(), previous)
    global_certificate(w)


@wp.kernel
def snapshot(
    rounds: wp.array[int],
    count: wp.array[int],
    boundary: wp.array[int],
    supported: wp.array[int],
    tets: wp.array2d[int],
    counts: wp.array[int],
    out_faces: wp.array2d[int],
    out_support: wp.array2d[int],
    out_tets: wp.array3d[int],
):
    i = wp.tid()
    r = rounds[0]
    if i == 0:
        counts[r] = count[0]
    if i < count[0]:
        out_faces[r, i] = boundary[i]
        out_support[r, i] = supported[i]
    if i < tets.shape[0]:
        for k in range(4):
            out_tets[r, i, k] = tets[i, k]


@pytest.mark.parametrize("failure,bit", [("nonfinite", 2), ("stack", 16)])
def test_support_kernel_reports_incomplete_traversal_like_original(failure, bit):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    seed, _, retained = seed_fixture(points, [[0, 1, 2, 3]])
    if failure == "nonfinite":
        bad = points.copy()
        bad[0, 0] = np.nan
        seed.points.assign(bad)
    else:
        # A deliberately malformed cyclic tree makes the bounded stack fill;
        # all accesses remain in range and both kernels must reject the batch.
        for field in [seed.tree.left, seed.tree.right]:
            data = field.numpy()
            data[1] = 1
            field.assign(data)
        data = seed.tree.count.numpy()
        data[1] = 0
        seed.tree.count.assign(data)
        for field, value in [(seed.tree.lower, -100), (seed.tree.upper, 100)]:
            data = field.numpy()
            data[1] = value
            field.assign(data)
    faces = wp.array(np.arange(4, dtype=np.int32), device=seed.points.device)
    count = wp.full(1, 4, dtype=int, device=seed.points.device)
    supported = wp.full(4, 1, dtype=int, device=seed.points.device)
    codes = []
    for kernel in [check_faces, record_face_support]:
        error = wp.zeros(1, dtype=int, device=seed.points.device)
        args = [seed.points, seed.source, faces, count, seed.tree, error, 0, 1]
        if kernel is record_face_support:
            args += [supported]
        wp.launch(kernel, 1, inputs=args, block_dim=32, device=seed.points.device)
        codes.append(int(error.numpy()[0]))
    assert codes[0] == codes[1] and codes[1] & bit
