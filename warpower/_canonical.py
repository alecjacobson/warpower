"""Deterministic ownership of exactly coincident sites."""

import warp as wp


@wp.func(inline=True)
def mix(value: wp.uint32):
    value = (value ^ (value >> wp.uint32(16))) * wp.uint32(0x85EBCA6B)
    value = (value ^ (value >> wp.uint32(13))) * wp.uint32(0xC2B2AE35)
    return value ^ (value >> wp.uint32(16))


@wp.kernel(enable_backward=False)
def insert(
    points: wp.array[wp.vec3],
    weights: wp.array[float],
    bits: wp.array2d[wp.uint32],
    table: wp.array[int],
    buckets: wp.array[int],
):
    i = wp.tid()
    h = wp.uint32(0)
    for axis in range(3):
        value = bits[i, axis]
        # Positive and negative zero describe the same position.
        if (value & wp.uint32(0x7FFFFFFF)) == wp.uint32(0):
            value = wp.uint32(0)
        h = mix(h ^ value)
    bucket = int(h & wp.uint32(table.shape[0] - 1))
    p = points[i]
    while True:
        previous = wp.atomic_cas(table, bucket, -1, i)
        if previous == -1:
            break
        other = points[previous]
        if p[0] == other[0] and p[1] == other[1] and p[2] == other[2]:
            if weights[i] > weights[previous] or (weights[i] == weights[previous] and i < previous):
                if wp.atomic_cas(table, bucket, previous, i) == previous:
                    break
            else:
                break
        else:
            bucket = (bucket + 1) & (table.shape[0] - 1)
    buckets[i] = bucket


@wp.kernel(enable_backward=False)
def resolve(table: wp.array[int], buckets: wp.array[int], owners: wp.array[int]):
    i = wp.tid()
    owners[i] = table[buckets[i]]


def canonicalize(points, weights):
    n = len(points)
    size = 1 << ((2 * n - 1).bit_length())
    table = wp.full(size, -1, dtype=int, device=points.device)
    buckets = wp.empty(n, dtype=int, device=points.device)
    owners = wp.empty(n, dtype=int, device=points.device)
    bits = wp.array(ptr=points.ptr, shape=(n, 3), dtype=wp.uint32, device=points.device)
    wp.launch(insert, n, inputs=[points, weights, bits, table, buckets], device=points.device)
    wp.launch(resolve, n, inputs=[table, buckets, owners], device=points.device)
    return owners, (table, buckets, bits)
