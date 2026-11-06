"""Minimal nested-loop sanitizer reproducer: no geometry or Warpower imports."""

import argparse

import warp as wp


@wp.kernel
def increment(value: wp.array[int], running: wp.array[int], limit: int):
    value[0] += 1
    running[0] = int(value[0] < limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--stack-bytes", type=int)
    parser.add_argument("--disable-mempool", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or not 0 <= args.depth <= 5:
        parser.error("iterations>=1 and depth in [0,5] required")
    if args.stack_bytes is not None or args.disable_mempool:
        from benchmarks.memcheck import configure

        configure(args.stack_bytes or 1024, args.disable_mempool)
    with wp.ScopedDevice("cuda:0"):
        value = wp.zeros(1, dtype=int)
        running = wp.ones_like(value)
        conditions = [wp.ones_like(value) for _ in range(args.depth)]
        wp.load_module(module=__name__, device="cuda:0")
        wp.synchronize_device("cuda:0")  # All allocations/fills complete before capture.

        def body(depth):
            if depth:
                conditions[depth - 1].fill_(1)

                def inner():
                    body(depth - 1)
                    conditions[depth - 1].zero_()

                wp.capture_while(conditions[depth - 1], inner)
            else:
                wp.launch(increment, 1, inputs=[value, running, args.iterations])

        with wp.ScopedCapture() as capture:
            wp.capture_while(running, lambda: body(args.depth))
        wp.capture_launch(capture.graph)
        result = int(value.numpy()[0])
        assert result == args.iterations
        print(f"Completed {result} iterations through {args.depth + 1} nested loops", flush=True)


if __name__ == "__main__":
    main()
