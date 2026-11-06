"""Standalone Warp sort-capture diagnostic; imports no Warpower geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from warp.utils import radix_sort_pairs

from benchmarks.memcheck import configure


@wp.kernel
def initialize(keys: wp.array[wp.uint64], values: wp.array[int], n: int):
    i = wp.tid()
    keys[i] = wp.uint64(n - 1 - i)
    values[i] = i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--disable-mempool", action="store_true")
    parser.add_argument("--stack-bytes", type=int, default=32768)
    parser.add_argument("--dot", type=Path)
    args = parser.parse_args()
    if args.n < 1 or not 0 <= args.depth <= 5:
        parser.error("n>=1 and depth in[0,5] required")
    configure(args.stack_bytes, args.disable_mempool)
    device = wp.get_device("cuda:0")
    keys = wp.empty(2 * args.n, dtype=wp.uint64, device=device)
    values = wp.empty(2 * args.n, dtype=int, device=device)
    condition = wp.ones(1, dtype=int, device=device)
    wp.load_module(module=__name__, device=device)
    wp.launch(initialize, args.n, inputs=[keys, values, args.n])
    radix_sort_pairs(keys, values, args.n, end_bit=63)
    wp.synchronize_device(device)

    def nested(depth):
        if depth:
            wp.capture_if(condition, lambda: nested(depth - 1))
        else:
            radix_sort_pairs(keys, values, args.n, end_bit=63)

    with wp.ScopedCapture() as cap:
        wp.launch(initialize, args.n, inputs=[keys, values, args.n])
        nested(args.depth)
    wp.synchronize_device(device)
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    for _ in range(3):
        wp.capture_launch(cap.graph)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(keys.numpy()[: args.n], np.arange(args.n, dtype=np.uint64))
    np.testing.assert_array_equal(
        values.numpy()[: args.n], np.arange(args.n - 1, -1, -1, dtype=np.int32)
    )
    print(
        json.dumps(dict(n=args.n, depth=args.depth, replays=3, exact_keys_and_payloads_match=True)),
        flush=True,
    )


if __name__ == "__main__":
    main()
