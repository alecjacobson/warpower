"""Observe active retry reasons and outcomes outside performance timing."""

import argparse
import json
from functools import wraps
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from warpower import _api

REASONS = {
    0: "success",
    1: "plane_or_output_capacity",
    2: "vertex_capacity",
    3: "hidden_or_duplicate",
    4: "queue_capacity",
    5: "clipping_boundary",
    6: "uncertain_predicate",
}


def histogram(values):
    """Count construction results by their specific failure reason."""
    codes, counts = np.unique(values, return_counts=True)
    return {REASONS[int(code)]: int(count) for code, count in zip(codes, counts)}


def observe(points, weights=None, **options):
    """Return an ordinary construction and synchronous per-stage observations.

    This diagnostic temporarily instruments the Python launch dispatcher in this
    process. Run it alone, outside CUDA capture and outside performance timing.
    Scheduled rows and eligible active cells are reported separately. The state
    filters match the production retry policy, including bounded FP64 recovery.
    """
    original_launch = wp.launch
    original_serial = _api.make_construct_kernel
    original_cooperative = _api.make_precision_kernel
    metadata = {}
    stages = []

    @wraps(original_serial)
    def serial(weighted=False, queue_capacity=64, robust=False, lane_stride=1, track_retries=True):
        kernel = original_serial(weighted, queue_capacity, robust, lane_stride, track_retries)
        metadata[kernel] = (False, queue_capacity, robust, lane_stride)
        return kernel

    @wraps(original_cooperative)
    def cooperative(weighted=False, queue_capacity=128, robust=True, lane_stride=32, capacity=128):
        kernel = original_cooperative(weighted, queue_capacity, robust, lane_stride, capacity)
        metadata[kernel] = (True, queue_capacity, robust, lane_stride)
        return kernel

    def launch(kernel, dim, *args, **kwargs):
        if kernel not in metadata:
            return original_launch(kernel, dim, *args, **kwargs)
        inputs = kwargs["inputs"]
        parallel, queue, robust, stride = metadata[kernel]
        start, scheduled = inputs[4], dim // stride
        if parallel:
            scheduled = max(0, min(scheduled, int(inputs[5].numpy()[0]) - start))
        owners = (
            inputs[3][start : start + scheduled].numpy()
            if scheduled
            else np.empty(0, dtype=np.int32)
        )
        initial = not parallel and not inputs[5]
        if parallel:
            rows = inputs[6].numpy()[owners] if inputs[8] and inputs[7] else owners
            scratch, statuses = inputs[9], inputs[12]
        else:
            rows = owners if initial else np.arange(start, start + scheduled)
            scratch, statuses = inputs[6], inputs[9]
        before = None if initial else statuses.numpy()[rows]
        active = np.ones(scheduled, dtype=bool)
        states = None
        if not initial:
            states = inputs[-1].numpy()[owners]
            flags = states[:, 0] | np.where(np.isin(before, [5, 6]), 1, 0)
            active = (
                ~np.isin(before, [0, 3])
                & ((flags & 4) == 0)
                & (((flags & 1) != 0) == robust)
                & (states[:, 1] <= scratch.capacity)
                & (states[:, 2] <= queue)
            )
            if parallel and robust:
                active &= ~(np.isin(before, [5, 6]) & ((flags & 2) != 0))
        result = original_launch(kernel, dim, *args, **kwargs)
        after = statuses.numpy()[rows]
        record = {
            "precision": "fp64" if robust else "fp32",
            "threads_per_cell": 32 if parallel else 1,
            "allocated_cell_capacity": scratch.capacity,
            "allocated_queue_capacity": queue,
            "scheduled_rows": scheduled,
            "active_cells": int(active.sum()),
            "reasons_before": None if initial else histogram(before[active]),
            "outcomes_after": histogram(after[active]),
            "skipped_outcomes_after": histogram(after[~active]),
        }
        if not initial:
            following = inputs[-1].numpy()[owners]
            record["active_site_ids"] = owners[active].tolist()
            record["requirements_before"] = states[active].tolist()
            record["requirements_after"] = following[active].tolist()
        stages.append(record)
        return result

    _api.make_construct_kernel = serial
    _api.make_precision_kernel = cooperative
    wp.launch = launch
    try:
        result = (
            warpower.voronoi_diagram(points, **options)
            if weights is None
            else warpower.power_diagram(points, weights, **options)
        )
    finally:
        wp.launch = original_launch
        _api.make_construct_kernel = original_serial
        _api.make_precision_kernel = original_cooperative
    return result, stages


def main():
    """Write untimed retry observations for a saved point array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()
    with np.load(args.input) as data:
        points = np.asarray(data["points"], np.float32)
        weights = np.asarray(data["weights"], np.float32) if args.weighted else None
    result, stages = observe(points, weights)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Untimed synchronous retry diagnostics",
                "stages": stages,
                "final_status": histogram(result.status.numpy()),
                "diagnostics": result.diagnostics,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
