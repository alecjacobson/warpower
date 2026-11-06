"""Independent exact separating-axis oracle for unshared triangle contact."""

import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp

from warpower._triangle_contact import triangles_intersect

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def contact_pairs(points: wp.array2d[wp.vec3], result: wp.array[int]):
    i = wp.tid()
    result[i] = int(
        triangles_intersect(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            points[i, 3],
            points[i, 4],
            points[i, 5],
        )
    )


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def exact_contact(points):
    # One common power-of-two denominator converts the *actual float32*
    # coordinates to arbitrary-precision integers without any rounding.
    ratios = [float(x).as_integer_ratio() for x in np.asarray(points, np.float32).flat]
    denominator = max(d for _, d in ratios)
    xyz = [n * (denominator // d) for n, d in ratios]
    left, right = (
        [tuple(xyz[i : i + 3]) for i in range(0, 9, 3)],
        [tuple(xyz[i : i + 3]) for i in range(9, 18, 3)],
    )
    a = [subtract(left[(i + 1) % 3], left[i]) for i in range(3)]
    b = [subtract(right[(i + 1) % 3], right[i]) for i in range(3)]
    na, nb = cross(a[0], a[1]), cross(b[0], b[1])
    assert any(na) and any(nb), "oracle requires nondegenerate triangles"
    # Face normals and all edge cross products suffice for noncoplanar pairs.
    # In-plane edge normals also separate coplanar triangles. Extra axes cannot
    # reject a real intersection: its point belongs to both projection ranges.
    axes = [na, nb] + [cross(x, y) for x in a for y in b]
    axes += [cross(na, x) for x in a] + [cross(nb, y) for y in b]
    for axis in axes:
        lp, rp = [dot(axis, p) for p in left], [dot(axis, p) for p in right]
        if max(lp) < min(rp) or max(rp) < min(lp):
            return False
    return True


def evaluate(pairs):
    source = wp.array(np.asarray(pairs, np.float32), dtype=wp.vec3, device="cuda:0")
    result = wp.empty(len(pairs), dtype=int, device="cuda:0")
    wp.launch(contact_pairs, len(pairs), inputs=[source, result])
    return result.numpy()


def test_imported_false_contacts_all_windings_pair_orders_and_scales():
    data = json.loads(
        (Path(__file__).parent / "data/triangle-contact-regressions.json").read_text()
    )
    pairs = []
    for item in data["pairs"]:
        points = np.asarray(item["points"], np.float32)
        assert not exact_contact(points)
        for left, right in itertools.product(itertools.permutations(range(3)), repeat=2):
            ids = list(left) + [x + 3 for x in right]
            for order in (ids, ids[3:] + ids[:3]):
                for exponent in (-100, -60, 0, 60, 100):
                    pairs.append(np.ldexp(points[order], exponent))
    np.testing.assert_array_equal(evaluate(pairs), 0)


def test_touching_crossing_coplanar_and_nearly_coplanar_against_integer_oracle():
    base = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    right = [
        [[0.25, 0.25, 0], [0.5, 0.25, 0], [0.25, 0.5, 0]],  # containment
        [[0.25, 0.25, -1], [0.25, 0.25, 1], [1, 1, 0]],  # crossing
        [[0, 0, 0], [-1, 0, 0], [0, -1, 0]],  # unindexed vertex touch
        [[0, 0, 0], [2, 0, 0], [0, -2, 0]],  # unindexed edge touch
        [[0.25, 0.25, 2**-40], [0.5, 0.25, 2**-40], [0.25, 0.5, 2**-40]],
        [[2, 2, 0], [3, 2, 0], [2, 3, 0]],  # coplanar disjoint, overlapping AABB
        [[1, 0, 0], [1, -1, 1], [1, -1, -1]],  # edge/vertex touch
        [[1, 0.5, 0], [1, -0.5, 0], [3, 0.5, 0]],  # coplanar edge crossing
    ]
    pairs = [np.array(base + r, np.float32) for r in right]
    np.testing.assert_array_equal([exact_contact(p) for p in pairs], [1, 1, 1, 1, 0, 0, 1, 1])
    rng = np.random.default_rng(92026)
    for _ in range(128):
        sample = rng.uniform(-1, 1, (6, 3)).astype(np.float32)
        pairs.append(sample)
        coplanar = sample.copy()
        coplanar[:, 2] = 0
        pairs.append(coplanar)
        almost = coplanar.copy()
        almost[:, 2] = np.ldexp(sample[:, 2], -40)
        pairs.append(almost)
    variants = []
    for p in pairs:
        for exponent in (-60, 0, 60):
            variants.append(np.ldexp(p, exponent))
            variants.append(np.ldexp(p[[4, 3, 5, 2, 1, 0]], exponent))
    expected = [int(exact_contact(p)) for p in variants]
    np.testing.assert_array_equal(evaluate(variants), expected)
