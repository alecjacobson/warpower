"""Prepare an unexecuted #175 follow-up; production source is untouched."""
import ast,hashlib,json
from pathlib import Path
w=Path('.work/vtm');src=Path('warpower/_stackless_seeds.py');s=src.read_text();m=ast.parse(s)
f=next(n for n in m.body if isinstance(n,ast.FunctionDef) and n.name=='check_stackless')
full=ast.get_source_segment(s,f)
full=full.replace('    segment: Tree,\n','    segment: Tree,\n    pending: wp.array[wp.uint32],\n')
exact=full.replace('def check_stackless(', 'def exact_check(').replace('    ids = wp.vec4i(', '    if (pending[i] & (wp.uint32(1) << wp.uint32(lane))) == wp.uint32(0):\n        return\n    ids = wp.vec4i(',1)
strict=full.replace('def check_stackless(', 'def strict_check(').replace('    node = int(1)','    uncertain = int(0)\n    node = int(1)',1)
start=strict.index('        result = symbolic_sphere(');end=strict.index('    if node != 0:',start)
strict=strict[:start]+'        uncertain = 1\n'+strict[end:]+'''\n    if uncertain != 0:
        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))
'''
header='''"""Private #175 follow-up: hierarchical strict queries and ordered exact lanes.

No runtime adoption. Preserve current bounded, Ball64 and symbolic predicates.
"""
import warp as wp
from warpower import _bounded_predicates as low
from warpower import _sphere_bounds as sb
from warpower._analytic_predicates import strict
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._prepared_seeds import PreparedSeedQueries, point_value, prepare
from warpower._stackless_seeds import successor

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
'''
body='''
class HierarchicalSplitQueries(PreparedSeedQueries):
    def __init__(self, *args, segment, **kwargs):
        super().__init__(*args, **kwargs)
        points, _, _, tree = self._check_inputs[:4]
        assert segment.n == len(points) == tree.n
        assert segment.ids.ptr == tree.ids.ptr and segment.leaf_size == 1
        assert segment.leaf_base == 1 << (len(points) - 1).bit_length()
        self.segment = segment
        self.pending = wp.zeros(self.capacity, dtype=wp.uint32, device=self.device)
        self._query_inputs = [*self._check_inputs, segment, self.pending]
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def run(self):
        self.pending.zero_()
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(strict_check, 32 * self.capacity, inputs=self._query_inputs,
                  block_dim=self.block_dim, device=self.device)
        wp.launch(exact_check, 32 * self.capacity, inputs=self._query_inputs,
                  block_dim=self.block_dim, device=self.device)


def install():
    from warpower import _stackless_seeds
    _stackless_seeds.StacklessSeedQueries = HierarchicalSplitQueries
'''
p=w/'hierarchical_split_v1.py';assert not p.exists();p.write_text(header+'\n\n@wp.kernel\n'+strict+'\n\n@wp.kernel\n'+exact+'\n'+body)
record=dict(scope=__doc__,baseline_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),stage='Prepared only, no compile/test/performance result; wait PR231 qualification and merge before GPU experiments.')
(w/'hierarchical-split-preparation-v1.json').write_text(json.dumps(record,indent=2)+'\n')
