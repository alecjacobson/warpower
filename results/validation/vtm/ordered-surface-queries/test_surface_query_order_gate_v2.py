"""Private ordered-query gate: unchanged scalar outputs across live count/domain/error changes."""
import numpy as np
import pytest
import warp as wp

from tests.test_domain import cube
from warpower import SurfaceDomain, _extraction
from warpower import _surface_query_cache as original
from surface_query_order_v4 import SizeGatedSurfaceQueryCache


@pytest.mark.parametrize('capacity', [64, 32768, 32769])
@pytest.mark.parametrize('mode', ['full', 'selection', 'retained'])
def test_live_size_gate_preserves_outputs(capacity, mode):
    surface, faces = cube()
    domain = SurfaceDomain(surface, faces)
    points = wp.array(np.array([[0, 0, 0], [.2, 0, 0], [0, .2, 0], [0, 0, .2]], np.float32), dtype=wp.vec3)
    cells = wp.array(np.tile(np.array([[0, 1, 2, 3]], np.int32), (capacity, 1)), dtype=int)
    count = wp.zeros(1, dtype=int)
    upstream = wp.zeros(1, dtype=int)
    caches = [original.SurfaceQueryCache(domain, capacity), SizeGatedSurfaceQueryCache(domain, capacity)]
    ordered = caches[1]
    assert ordered._has_order_storage == (capacity > 32768)
    if capacity <= 32768:
        assert ordered.order_keys is ordered.order_ids is ordered.order_enabled is None
    kernel = {'full': original.classify_cached, 'selection': original.selection_cached, 'retained': original.retained_cached}[mode]
    outputs, graphs = [], []
    stream = wp.Stream(device=domain.device)
    for cache in caches:
        error = wp.zeros(1, dtype=int)
        keep = wp.full(capacity, 77, dtype=int)
        quality = wp.full(capacity, 77, dtype=wp.float64)
        arrays = dict(error=error, keep=keep, quality=quality)
        args = [domain.tree, points, cells, count, .4, error]
        if mode == 'retained':
            args += [keep, quality]
        else:
            center = wp.full(capacity, wp.vec3d(77), dtype=wp.vec3d)
            radius = wp.full(capacity, 77, dtype=wp.float64)
            distance = wp.full(capacity, 77, dtype=wp.float64)
            volume = wp.full(capacity, 77, dtype=wp.float64)
            arrays.update(center=center, radius=radius, distance=distance, volume=volume)
            args += [center, radius, distance, keep, volume, quality]
        wp.synchronize_device(domain.device)
        with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream) as capture:
            domain.refit()
            wp.launch(_extraction.input_ready, 1, inputs=[domain.error, upstream, count, capacity, error], stream=stream)
            cache.launch(kernel, capacity, args)
        outputs.append(arrays)
        graphs.append(capture.graph)
    addresses = [[array.ptr for array in row.values()] for row in outputs]
    phases = [(min(capacity-1, 32767), 0, False), (capacity, 0, False), (capacity, 0, False), (0, 0, False), (capacity+1, 0, False), (capacity, 4, False), (capacity, 0, True), (capacity, 0, False)]
    for active, failure, invalid_domain in phases:
        current_surface = surface.copy()
        if invalid_domain:
            current_surface[0] = np.nan
        domain.points.assign(current_surface)
        count.fill_(active)
        upstream.fill_(failure)
        wp.synchronize_device(domain.device)
        for graph in graphs:
            wp.capture_launch(graph, stream=stream)
        wp.synchronize_stream(stream)
        for name in outputs[0]:
            a, b = outputs[0][name].numpy(), outputs[1][name].numpy()
            assert a.tobytes() == b.tobytes(), (mode, active, failure, invalid_domain, name)
        error = int(outputs[1]['error'].numpy()[0])
        assert (error != 0) == (active > capacity or failure != 0 or invalid_domain)
        if ordered._has_order_storage:
            enabled = int(ordered.order_enabled.numpy()[0])
            assert enabled == int(error == 0 and active >= 32768)
            if enabled:
                np.testing.assert_array_equal(np.sort(ordered.order_ids.numpy()[:capacity]), np.arange(capacity))
        assert addresses == [[array.ptr for array in row.values()] for row in outputs]
