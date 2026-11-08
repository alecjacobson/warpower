"""Exact rational AABB support checks for the private coordinate-plane predicate."""

import itertools
from fractions import Fraction as F

import numpy as np
import warp as wp

from warpower._proved_completion import axis_supported

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def evaluate(data: wp.array2d[wp.vec3], output: wp.array[int]):
    i = wp.tid()
    output[i] = int(
        axis_supported(data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5])
    )


def orient(a, b, c, d):
    q = [[F(float(x)) for x in v] for v in [a, b, c, d]]
    u, v, w = [[q[i][k] - q[0][k] for k in range(3)] for i in range(1, 4)]
    return (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )


def test_axis_supported_matches_rational_box_oracle():
    rows = []
    expected = []
    names = []
    valid = []
    for exponent, translation, axis, side in itertools.product(
        [-140, -120, -60, 0, 60, 120], [-4, -1, 0, 1, 4], range(3), [-1, 1]
    ):
        s = np.float32(2.0**exponent)
        plane = np.float32(translation * s)
        j = (axis + 1) % 3
        k = (axis + 2) % 3
        a = np.zeros(3, np.float32)
        a[axis] = plane
        b = a.copy()
        b[j] = s
        c = a.copy()
        c[k] = s
        d = a.copy()
        d[j] = s * np.float32(0.25)
        d[k] = s * np.float32(0.25)
        d[axis] = plane + side * s
        if orient(a, b, c, d) > 0:
            b, c = c, b
        assert orient(a, b, c, d) < 0
        lo = np.minimum.reduce([a, b, c, d]) - s
        hi = np.maximum.reduce([a, b, c, d]) + s
        if side > 0:
            lo[axis] = plane
        else:
            hi[axis] = plane
        for mode in ["on", "inside", "outside", "near_axis", "flat_opposite"]:
            data = np.asarray([a, b, c, d, lo, hi], np.float32).copy()
            if mode == "inside":
                data[4 if side > 0 else 5, axis] = plane + side * s * np.float32(0.25)
            elif mode == "outside":
                data[4 if side > 0 else 5, axis] = np.nextafter(
                    plane, np.float32(-np.inf if side > 0 else np.inf)
                )
            elif mode == "near_axis":
                data[1, axis] = np.nextafter(plane, np.float32(np.inf if side > 0 else -np.inf))
            elif mode == "flat_opposite":
                data[3, axis] = plane
            rows.append(data)
            expected.append(mode in ["on", "inside"])
            names.append([exponent, translation, axis, side, mode])
            valid.append(orient(*data[:4]) < 0)
    # Signed-zero equality is geometric equality, including all supported axes.
    for axis, side in itertools.product(range(3), [-1, 1]):
        index = names.index([0, 0, axis, side, "on"])
        data = rows[index].copy()
        data[1, axis] = np.float32(-0.0)
        rows.append(data)
        expected.append(True)
        names.append(["signed_zero", axis, side])
        valid.append(True)
    # Explicitly unsupported scalar preconditions must not produce a shortcut.
    base = rows[names.index([0, 1, 2, 1, "on"])].copy()
    for label, which, value in [
        ("nan_plane", 0, np.nan),
        ("inf_opposite", 3, np.inf),
        ("nan_lower", 4, np.nan),
        ("inf_upper", 5, np.inf),
    ]:
        data = base.copy()
        if which == 0:
            data[:3, 2] = value
        else:
            data[which, 2] = value
        rows.append(data)
        expected.append(False)
        names.append([label])
        valid.append(False)
    data = base.copy()
    data[4, 2] = data[5, 2] + 1
    rows.append(data)
    expected.append(False)
    names.append(["reversed_box"])
    valid.append(False)
    rng = np.random.default_rng(214)
    for i in range(128):
        p = rng.normal(size=(4, 3)).astype(np.float32)
        if orient(*p) > 0:
            p[[1, 2]] = p[[2, 1]]
        assert orient(*p) < 0
        data = np.concatenate([p, [p.min(axis=0) - 1, p.max(axis=0) + 1]])
        rows.append(data)
        expected.append(False)
        names.append(["general_face", i])
        valid.append(True)
    data = np.asarray(rows, np.float32)
    expected = np.asarray(expected, np.int32)
    oracle = []
    for row, e, ok in zip(data, expected, valid):
        if not ok:
            oracle.append(None)
            continue
        signs = [orient(*row[:3], q) for q in itertools.product(*zip(row[4], row[5]))]
        all_inside = all(s <= 0 for s in signs)
        oracle.append(all_inside)
        if e:
            assert all_inside
        if not e and np.all(row[0] == row[1]):
            raise AssertionError("unexpected degenerate face")
    wp.init()
    wp.set_device("cuda:0")
    gpu = wp.array(data, dtype=wp.vec3)
    out = wp.zeros(len(data), dtype=int)
    with wp.ScopedCapture() as capture:
        wp.launch(evaluate, len(data), inputs=[gpu, out])
    wp.capture_launch(capture.graph)
    actual = out.numpy()
    mismatch = np.flatnonzero(actual != expected).tolist()
    assert not mismatch, [
        (i, names[i], data[i].tolist(), int(actual[i]), int(expected[i])) for i in mismatch
    ]


@wp.kernel
def mark_faces(count: wp.array[int], observed: wp.array[int]):
    observed[0] += count[0]


def test_certificate_rejects_invalid_cells_before_axis_shortcut():
    from warpower._canonical import canonicalize
    from warpower._proved_completion import ProvedCertificate
    from warpower._site_bvh import SiteHierarchy

    points = np.array([[8, -4, 2], [9, -4, 2], [8, -3, 2], [8, -4, 3]], np.float32)
    p = wp.array(points, dtype=wp.vec3, device="cuda:0")
    t = wp.array([[0, 1, 2, 3]], dtype=int, device=p.device)
    count = wp.ones(1, dtype=int, device=p.device)
    hierarchy = SiteHierarchy(p)
    owners, retained = canonicalize(p, wp.zeros(len(p), dtype=float, device=p.device))
    hierarchy.tree.owners = owners
    certificate = ProvedCertificate(p, t, count, hierarchy.tree)
    certificate._input_proved.zero_()
    observed = wp.zeros(1, dtype=int, device=p.device)
    original = certificate._hull_support

    def checked(tree):
        wp.launch(mark_faces, 1, inputs=[certificate._boundary_count, observed], device=p.device)
        original(tree)

    certificate._hull_support = checked
    wp.load_module(module=__name__, device=p.device)
    with wp.ScopedCapture() as capture:
        observed.zero_()
        certificate.run()
    for kind in ["valid", "inverted", "nonfinite", "empty", "overcount", "flat", "restored"]:
        xyz = points.copy()
        cells = np.array([[0, 1, 2, 3]], np.int32)
        n = 1
        if kind == "inverted":
            cells[0, [0, 1]] = cells[0, [1, 0]]
        elif kind == "nonfinite":
            xyz[0, 0] = np.nan
        elif kind == "empty":
            n = 0
        elif kind == "overcount":
            n = 2
        elif kind == "flat":
            xyz[3, 2] = 2
        p.assign(xyz)
        t.assign(cells)
        count.fill_(n)
        # Invalid frames exercise rejection before a current hierarchy can be used;
        # valid/restored frames use exactly its original, finite coordinates.
        wp.capture_launch(capture.graph)
        good = kind in ["valid", "restored"]
        assert (int(certificate.error.numpy()[0]) == 0) == good
        assert int(certificate.tetrahedra_count.numpy()[0]) == int(good)
        assert int(observed.numpy()[0]) == (4 if good else 0)
