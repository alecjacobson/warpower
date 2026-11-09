"""Bounded observer buffers never change solver inputs or trust stale diagnostics."""

import os

import numpy as np
import warp as wp

from benchmarks.selective_failures import COLUMNS, Inputs, begin_call, record

DEVICE = os.environ.get("WARPOWER_DIAGNOSTIC_TEST_DEVICE", "cuda:0")


def inputs():
    s = Inputs()
    for name in [
        "mode",
        "iteration",
        "halving",
        "active",
        "full_count",
        "recovered",
        "accepted",
        "rejections",
        "selection_error",
        "reason",
        "retained",
        "extraction_error",
        "valid",
        "diagnostics",
        "extraction_count",
        "additions",
        "rounds",
        "mesh_error",
        "boundary_error",
    ]:
        setattr(s, name, wp.zeros(1, dtype=int, device=DEVICE))
    for name in ["alpha", "minimum", "previous_minimum"]:
        setattr(s, name, wp.ones(1, dtype=wp.float64, device=DEVICE))
    s.mode.fill_(1)
    s.recovered.fill_(1)
    s.rejections.fill_(1)
    s.selection_error.fill_(16)
    s.reason.fill_(3)
    s.diagnostics.fill_(2)
    s.extraction_count.fill_(32)
    return s


def test_bounded_event_and_failure_buffers_keep_original_error_attribution():
    s = inputs()
    state = wp.zeros(8, dtype=int, device=DEVICE)
    rows = wp.full((2, len(COLUMNS)), -9, dtype=int, device=DEVICE)
    values = wp.zeros((2, 3), dtype=wp.float64, device=DEVICE)
    for _ in range(3):
        wp.launch(begin_call, 1, inputs=[state], device=DEVICE)
        wp.launch(record, 1, inputs=[s, state, rows, values, 1], device=DEVICE)
    a, b = rows.numpy()
    assert a[8] == 1 and a[9] == 16 and a[12] == 0 and a[13] == 0 and a[14] == 2
    assert a[15] == 32 and a[-1] == 0 and b[-1] == -1
    assert state.numpy()[5] == state.numpy()[6] == 1
    assert state.numpy()[7] == -1
    assert s.selection_error.numpy()[0] == 16 and s.diagnostics.numpy()[0] == 2
    np.testing.assert_array_equal(values.numpy(), 1)


def test_stale_or_ordinary_extraction_fields_are_unavailable():
    s = inputs()
    state = wp.zeros(8, dtype=int, device=DEVICE)
    rows = wp.full((2, len(COLUMNS)), -9, dtype=int, device=DEVICE)
    values = wp.zeros((2, 3), dtype=wp.float64, device=DEVICE)
    wp.launch(begin_call, 1, inputs=[state], device=DEVICE)
    s.rejections.zero_()
    wp.launch(record, 1, inputs=[s, state, rows, values, 1], device=DEVICE)
    s.rejections.fill_(1)
    s.recovered.zero_()
    wp.launch(record, 1, inputs=[s, state, rows, values, 1], device=DEVICE)
    observed = rows.numpy()
    np.testing.assert_array_equal(observed[:, 12:18], -1)
    np.testing.assert_array_equal(observed[:, -1], -1)
    assert observed[:, 8].tolist() == [0, 1]
    assert state.numpy()[4] == 0
