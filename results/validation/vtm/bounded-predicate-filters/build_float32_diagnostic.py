"""Instrument both private consumers; timings from this variant are excluded."""
import ast
from pathlib import Path
w=Path('.work/vtm')
face=(w/'float32_face_search.py').read_text()
face=face.replace('import bounded_float32_filters as af','import bounded_float32_filters as af\nfrom warpower import _analytic_predicates as old\nfrom warpower import _filtered_predicates as fp')
face=face.replace('def search_float32(', 'def counted_search_float32(').replace('    statistics: wp.array[int],','    statistics: wp.array[int],\n    counters: wp.array[wp.int64],')
face=face.replace('    best = int(-1)','    oa, of, od, sa, sf, sd = int(0), int(0), int(0), int(0), int(0), int(0)\n    best = int(-1)')
face=face.replace('        orientation = af.orient3d(a, b, c, points[q])','''        oa += 1
        value = af.orientation_filter(a, b, c, points[q])
        orientation = fp.predicate(value)
        if not old.strict(value):
            of += 1
            value = old.orientation_filter(a, b, c, points[q])
            orientation = fp.predicate(value)
            if not old.strict(value):
                od += 1
                orientation = fp.orient3d(a, b, c, points[q])''')
face=face.replace('            result = af.symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)','''            sa += 1
            ids = wp.vec4i(face[0], face[1], face[2], best)
            value = af.sphere_filter(a, b, c, points[best], points[q])
            result = fp.predicate(value)
            if not old.strict(value):
                sf += 1
                value = old.sphere_filter(a, b, c, points[best], points[q])
                result = fp.predicate(value)
                if not old.strict(value):
                    sd += 1
                    result = old.legacy_symbolic_sphere(points, ids, q)''')
# An invalid result is never expected in this diagnostic trajectory. Record an
# explicit incomplete-counter flag rather than silently reporting partial counts.
face=face.replace('wp.atomic_or(error, 0, 2)', 'wp.atomic_or(error, 0, 2)\n'+' '*12+'wp.atomic_add(counters, 6, wp.int64(1))',1)
# Second error has a deeper indentation.
pos=face.index('            if result.valid')
face=face[:pos]+face[pos:].replace('wp.atomic_or(error, 0, 2)','wp.atomic_or(error, 0, 2)\n'+' '*16+'wp.atomic_add(counters, 6, wp.int64(1))',1)
face += '\n'.join('    wp.atomic_add(counters, '+str(i)+', wp.int64('+name+'))' for i,name in enumerate(['oa','of','od','sa','sf','sd']))+'\n'
seed=(w/'float32_seed_queries.py').read_text().split('class Float32GroupedQueries')[0]
seed=seed[seed.index('@wp.kernel'):].replace('def check_float32_groups(', 'def counted_float32_groups(').replace('    upper: wp.array[wp.vec3],','    upper: wp.array[wp.vec3],\n    counters: wp.array[wp.int64],')
seed=seed.replace('    for group in range(lower.shape[0]):','    attempted, fallback, exact = int(0), int(0), int(0)\n    for group in range(lower.shape[0]):')
seed=seed.replace('        fast = low.sphere_filter','        attempted += 1\n        fast = low.sphere_filter')
seed=seed.replace('        value = point_value(shape, p)','        fallback += 1\n        value = point_value(shape, p)')
seed=seed.replace('        result = symbolic_sphere(points, ids, q)','        exact += 1\n        result = symbolic_sphere(points, ids, q)')
seed+='\n'.join('    wp.atomic_add(counters, '+str(i)+', wp.int64('+name+'))' for i,name in enumerate(['attempted','fallback','exact']))+'\n'
header='''\nfrom warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._analytic_predicates import strict
from warpower._prepared_seeds import point_value, prepare
from warpower._grouped_seeds import GroupedSeedQueries, group_bounds
import bounded_float32_filters as low
'''
tail='''
face_instances, seed_instances = [], []
class CountedQueries(GroupedSeedQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters = wp.zeros(3, dtype=wp.int64, device=self.device)
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)
        seed_instances.append(self)
    def run(self):
        wp.launch(group_bounds, len(self.lower), inputs=self._group_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(counted_float32_groups, 32*self.capacity, inputs=[*self._query_inputs,self.counters], block_dim=self.block_dim, device=self.device)

def install():
    from warpower import _grouped_seeds
    from warpower._proved_completion import SeededFacetCompletion
    _grouped_seeds.GroupedSeedQueries = CountedQueries
    original = SeededFacetCompletion.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self._float32_counters=wp.zeros(7,dtype=wp.int64,device=self.device)
        face_instances.append(self)
    def search(self,tree):
        c=self._certificate
        self._launch(counted_search_float32,4*self.capacity*self._workers,[self.points,tree.owners,self.tetrahedra,c._boundary,c._boundary_count,self._candidates,self._workers,self.error,c._face_supported,c.face_statistics,self._float32_counters],block_dim=self._query_block_dim)
    SeededFacetCompletion.__init__=init
    SeededFacetCompletion._search_faces=search
    for block in [32,256]:
        wp.load_module(module=__name__,device='cuda:0',block_dim=block)

if __name__=='__main__':
    import json,os
    from pathlib import Path
    install()
    from benchmarks.mesher import main
    main()
    face_names=['orientations','orientation_f32_uncertain','orientation_f64_uncertain','spheres','sphere_f32_uncertain','sphere_f64_uncertain','invalid_incomplete_threads']
    seed_names=['spheres','sphere_f32_uncertain','prepared_ball64_uncertain']
    rows=dict(scope='Instrumented complete graph executions; timings excluded.',face=[dict(site_capacity=len(x.points),counts=dict(zip(face_names,map(int,x._float32_counters.numpy())))) for x in face_instances],seed=[dict(site_capacity=len(x._check_inputs[0]),counts=dict(zip(seed_names,map(int,x.counters.numpy())))) for x in seed_instances])
    assert all(x['counts']['invalid_incomplete_threads']==0 for x in rows['face'])
    Path(os.environ['WARPOWER_FLOAT32_DIAGNOSTIC']).write_text(json.dumps(rows,indent=2)+'\\n')
'''
source=face+header+seed+tail
ast.parse(source)
(w/'float32_filter_counters.py').write_text(source)
