"""Private #185: replace exactly one leaf predicate in the current kernel."""
import ast
from pathlib import Path

w=Path('.work/vtm')
s=Path('warpower/_proved_completion.py').read_text()
start=s.index('@wp.kernel\ndef record_face_support(')
stop=s.index('@wp.kernel\ndef search_unproved_faces(',start)
kernel=s[start:stop]
old='result = fp.orient3d(a, b, c, points[q])'
assert kernel.count(old)==1
modified=kernel.replace(old,'result = af.orient3d(a, b, c, points[q])')
assert ast.dump(ast.parse(modified.replace('result = af.orient3d','result = fp.orient3d'))) == ast.dump(ast.parse(kernel))
header='''"""Private bounded orientation in hull-support leaf queries only."""
import warp as wp
from warpower import _bounded_predicates as af
from warpower import _filtered_predicates as fp
from warpower._bvh import Tree
from warpower._mesh_topology import face_vertices
from warpower._delaunay_certificate import halfspace, inside_halfspace
import warpower._proved_completion as pc

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

'''
footer='''
def install():
    pc.record_face_support=record_face_support
    original_init=pc.ProvedCertificate.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        wp.load_module(module=__name__,device=self.device)
    pc.ProvedCertificate.__init__=init
'''
out=w/'bounded_hull_support.py'
assert not out.exists();out.write_text(header+modified+footer)
(w/'test-bounded-hull-support-existing.py').write_text('''import pytest
from bounded_hull_support import install
install()
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_large_reuse.py','tests/test_support_reuse.py']))
''')
(w/'run-bounded-hull-support.py').write_text('''from bounded_hull_support import install
install()
from benchmarks.mesher import main
main()
''')
