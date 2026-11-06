"""Run pytest with an explicit CUDA stack allowance for instrumented graphs.

Diagnostic setup only: the Warpower runtime never imports this module. Native
driver calls configure the test process, not the geometry or graph replay.
"""

import argparse
import ctypes
import json

import warp as wp


def configure(stack_bytes=32768, disable_mempool=False):
    device = wp.get_device("cuda:0")
    device.make_current()
    if disable_mempool:
        wp.set_mempool_enabled(device, False)
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    driver.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    before, after = ctypes.c_size_t(), ctypes.c_size_t()
    if driver.cuCtxGetLimit(ctypes.byref(before), 0) != 0:
        raise RuntimeError("Could not read CUDA stack limit")
    if driver.cuCtxSetLimit(0, max(before.value, stack_bytes)) != 0:
        raise RuntimeError("Could not set CUDA stack limit")
    if driver.cuCtxGetLimit(ctypes.byref(after), 0) != 0:
        raise RuntimeError("Could not verify CUDA stack limit")
    print(
        json.dumps(
            {
                "diagnostic_cuda_stack_bytes_before": before.value,
                "diagnostic_cuda_stack_bytes_after": after.value,
                "diagnostic_mempool_enabled": wp.is_mempool_enabled(device),
            }
        ),
        flush=True,
    )


def main():
    import pytest

    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-bytes", type=int, default=32768)
    parser.add_argument("--disable-mempool", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stack_bytes < 1024:
        parser.error("stack-bytes must be at least 1024")
    configure(args.stack_bytes, args.disable_mempool)
    tests = args.pytest_args
    if tests[:1] == ["--"]:
        tests = tests[1:]
    raise SystemExit(pytest.main(tests))


if __name__ == "__main__":
    main()
