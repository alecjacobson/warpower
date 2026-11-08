"""Integrate the measured ordering policy without changing original scalar cache kernels."""
import ast
import hashlib
import json
from pathlib import Path

w = Path('.work/vtm')
proof = json.loads((w/'surface-query-order-pilot-v2-independent.json').read_text())
assert proof['complete'] and proof['exact_geometry_transfers'] == 8
for name, row in proof['cases'].items():
    assert all(row['work_equal'].values())
    d = row['comparison']
    assert all(d[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
    if name != 'thin-hand-seed7':
        assert d['points_bitwise_equal']
    else:
        assert d['maximum_coordinate_difference'] < 1e-16
        assert d['maximum_stored_quality_difference'] < 1e-14
        assert d['minimum_quality_before'] == d['minimum_quality_after']
contract = json.loads((w/'surface-query-order-contracts-v4.json').read_text())
assert contract['passed'] and contract['exit_code'] == 0
for path, digest in contract['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
old = ast.parse((w/'surface_query_order_v1.py').read_text())
gate = ast.parse((w/'surface_query_order_v4.py').read_text())
names = ['centroid_keys','classify_cached_ordered','selection_cached_ordered','retained_cached_ordered']
functions = [next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name == name) for name in names]
functions.append(next(n for n in gate.body if isinstance(n,ast.FunctionDef) and n.name == 'choose_order'))
header = '''"""Spatial execution ordering for large private mesher surface-query batches.

Morton centroids select a permutation only. Original scalar query arithmetic,
cell IDs, outputs, cache validity and error handling are retained. Small caches
use the original launch path and own no ordering storage.
"""

import warp as wp
from warp.utils import radix_sort_pairs

from . import _surface_query_cache as original
from ._bvh import expand_bits
from ._surface_bvh import SurfaceTree
from ._surface_query_cache import (
    QueryState,
    SurfaceQueryCache as BaseCache,
    cached_nearest,
    cached_side,
    selection_center_cached,
)
from ._wide_center import exact_geometry, uncertain

wp.set_module_options({"enable_backward": False, "fast_math": False})

'''
body = '\n\n'.join(ast.unparse(n) for n in functions)
host = '''


ORDERED = {
    original.classify_cached: classify_cached_ordered,
    original.selection_cached: selection_cached_ordered,
    original.retained_cached: retained_cached_ordered,
}


class OrderedSurfaceQueryCache(BaseCache):
    """Keep sorting optional and all live batch decisions on device."""

    _minimum_query_count = 32768

    def __init__(self, domain, capacity):
        self.minimum_query_count = self._minimum_query_count
        self._has_order_storage = capacity > self.minimum_query_count
        super().__init__(domain, capacity)
        self.order_keys = self.order_ids = self.order_enabled = None
        if self._has_order_storage:
            with wp.ScopedDevice(self.device):
                self.order_keys = wp.zeros(2 * capacity, dtype=wp.uint64)
                self.order_ids = wp.zeros(2 * capacity, dtype=int)
                self.order_enabled = wp.ones(1, dtype=int)
                for block in [32, 256]:
                    wp.load_module(module=__name__, device=self.device, block_dim=block)
                # Initialize the sort outside capture; each conditional capture
                # then owns its retained CUB scratch through Warp's child graph.
                self._sort()
            wp.synchronize_device(self.device)

    def _sort(self):
        radix_sort_pairs(self.order_keys, self.order_ids, self.capacity, end_bit=64)

    def _ordered(self, kernel, dim, args, block_dim):
        wp.launch(
            centroid_keys, self.capacity,
            inputs=[args[0], args[1], args[2], args[3], args[5], self.order_keys, self.order_ids],
            device=self.device, block_dim=256,
        )
        self._sort()
        wp.launch(
            ORDERED[kernel], dim,
            inputs=[*args, self.current, self.previous, self.table, self.order_ids],
            device=self.device, block_dim=block_dim,
        )
        self.after(args[5])

    def launch(self, kernel, dim, args, block_dim=32):
        if not self._has_order_storage:
            return BaseCache.launch(self, kernel, dim, args, block_dim)
        assert kernel in ORDERED and dim == self.capacity
        if wp.get_stream(self.device).is_capturing:
            wp.launch(
                choose_order, 1,
                inputs=[args[3], args[5], self.minimum_query_count, self.order_enabled],
                device=self.device, block_dim=32,
            )
            wp.capture_if(
                self.order_enabled,
                lambda: self._ordered(kernel, dim, args, block_dim),
                lambda: BaseCache.launch(self, kernel, dim, args, block_dim),
            )
        else:
            self._ordered(kernel, dim, args, block_dim)
'''
p = Path('warpower/_ordered_surface_queries.py')
assert not p.exists()
p.write_text(header + body + host)
for name in ['_extraction.py','_quality_moves.py','_selective.py']:
    p = Path('warpower')/name
    s = p.read_text()
    before = 'from ._surface_query_cache import SurfaceQueryCache'
    assert s.count(before) == 1
    p.write_text(s.replace(before, 'from ._ordered_surface_queries import OrderedSurfaceQueryCache as SurfaceQueryCache'))
p = Path('tests/test_ordered_surface_queries.py')
assert not p.exists()
s = (w/'test_surface_query_order_gate_v2.py').read_text().replace('from surface_query_order_v4 import SizeGatedSurfaceQueryCache', 'from warpower._ordered_surface_queries import OrderedSurfaceQueryCache').replace('SizeGatedSurfaceQueryCache','OrderedSurfaceQueryCache')
s = s.replace('"""Private ordered-query gate: unchanged scalar outputs across live count/domain/error changes."""', '"""Ordered query parity across scale, cache state and live device batch/error changes."""')
s += '''


class ForcedOrderedCache(OrderedSurfaceQueryCache):
    # Exercise the actual ordered kernels on the existing small geometric fixtures.
    _minimum_query_count = 0


@pytest.mark.parametrize("mode", ["full", "selection", "retained"])
@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_ordered_cold_warm_classifiers_match_ordinary(monkeypatch, mode, scale):
    from tests import test_surface_query_cache as contracts

    monkeypatch.setattr(contracts, "Cache", ForcedOrderedCache)
    contracts.test_cold_warm_classification_matches_ordinary(mode, scale)


def test_ordered_cache_preserves_epoch_failure_and_saturation_contract(monkeypatch):
    from tests import test_surface_query_cache as contracts

    monkeypatch.setattr(contracts, "Cache", ForcedOrderedCache)
    contracts.test_captured_queries_refit_cutoff_failure_recovery_and_saturation()
'''
p.write_text(s)
print('Prepared production module, three private constructor imports and nineteen focused contracts.')
