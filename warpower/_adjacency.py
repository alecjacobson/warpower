"""Convert raw cell neighbors into symmetric CSR without host geometry work."""

import warp as wp
from warp.utils import array_scan

wp.set_module_options({"enable_backward": False})


@wp.func(inline=True)
def neighbor_at(
    i: int, j: int, base: wp.array2d[int], retry: wp.array2d[int], source: wp.array[int]
):
    row = source[i]
    if row >= 0:
        return retry[row, j]
    return base[i, j]


@wp.func(inline=True)
def contains(
    i: int,
    target: int,
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    source: wp.array[int],
):
    for k in range(counts[i]):
        if neighbor_at(i, k, base, retry, source) == target:
            return True
    return False


@wp.kernel(enable_backward=False)
def mark_asymmetric(
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    source: wp.array[int],
    flags: wp.array[int],
):
    i = wp.tid()
    for k in range(counts[i]):
        j = neighbor_at(i, k, base, retry, source)
        if not contains(j, i, counts, base, retry, source):
            wp.atomic_max(flags, i, 1)
            wp.atomic_max(flags, j, 1)


@wp.kernel(enable_backward=False)
def gather_marked(flags: wp.array[int], jobs: wp.array[int], count: wp.array[int]):
    i = wp.tid()
    if flags[i] != 0:
        at = wp.atomic_add(count, 0, 1)
        jobs[at] = i


@wp.kernel(enable_backward=False)
def gather_repair_seeds(
    jobs: wp.array[int],
    first: int,
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    source: wp.array[int],
    seeds: wp.array2d[int],
):
    slot, k = wp.tid()
    job = first + slot
    i = jobs[job]
    j = int(-1)
    if k < counts[i]:
        j = neighbor_at(i, k, base, retry, source)
    seeds[job, k] = j


def asymmetric_cells(neighbors, retry, source, counts):
    n, device = len(counts), counts.device
    flags = wp.zeros(n, dtype=int, device=device)
    jobs = wp.empty(n, dtype=int, device=device)
    count = wp.zeros(1, dtype=int, device=device)
    wp.launch(mark_asymmetric, n, inputs=[counts, neighbors, retry, source, flags], device=device)
    wp.launch(gather_marked, n, inputs=[flags, jobs, count], device=device)
    return jobs, int(count.numpy()[0])


@wp.kernel(enable_backward=False)
def count_symmetric(
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    source: wp.array[int],
    sizes: wp.array[int],
):
    i = wp.tid()
    for k in range(counts[i]):
        j = neighbor_at(i, k, base, retry, source)
        if counts[j] > 0:
            wp.atomic_add(sizes, i + 1, 1)
            if not contains(j, i, counts, base, retry, source):
                wp.atomic_add(sizes, j + 1, 1)


@wp.kernel(enable_backward=False)
def fill_symmetric(
    counts: wp.array[int],
    base: wp.array2d[int],
    retry: wp.array2d[int],
    source: wp.array[int],
    heads: wp.array[int],
    adjacency: wp.array[int],
):
    i = wp.tid()
    for k in range(counts[i]):
        j = neighbor_at(i, k, base, retry, source)
        if counts[j] > 0:
            pos = wp.atomic_add(heads, i, 1)
            adjacency[pos] = j
            if not contains(j, i, counts, base, retry, source):
                pos = wp.atomic_add(heads, j, 1)
                adjacency[pos] = i


def compact(neighbors, retry, source, counts):
    n, device = len(counts), counts.device
    sizes = wp.zeros(n + 1, dtype=int, device=device)
    offsets = wp.empty(n + 1, dtype=int, device=device)
    wp.launch(count_symmetric, n, inputs=[counts, neighbors, retry, source, sizes], device=device)
    array_scan(sizes, offsets)
    # Only the final count crosses to the host; all adjacency processing is GPU.
    total = int(offsets[n : n + 1].numpy()[0])
    adjacency = wp.empty(total, dtype=int, device=device)
    heads = wp.clone(offsets)
    wp.launch(
        fill_symmetric,
        n,
        inputs=[counts, neighbors, retry, source, heads, adjacency],
        device=device,
    )
    return adjacency, offsets
