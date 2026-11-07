"""Integrate the validated private filters without touching standalone kernels."""
import json
from pathlib import Path
w=Path('.work/vtm')
assert len(json.loads((w/'analytic-face-broad-independent.json').read_text()))==6
old_path=Path('warpower/_proved_completion.py');old=old_path.read_text()
start=old.index('@wp.kernel\ndef search_unproved_faces(')
end=old.index('\n\nclass ProvedCertificate',start)
reference=old[start:end]
source=(w/'analytic_face_search.py').read_text().split('\n\n@wp.kernel')[0]+'\n'
source=source.replace('Private #166 analytic first-stage filters; default production is untouched.','Analytic first-stage filters for sign-only VTM facet searches.')
source=source.replace('Grouping, non-fused arithmetic, and strict sign acceptance are essential.','Calling kernels require fast_math=False and fuse_fp=False. Grouping and strict\nnonzero acceptance are essential. Uncertain signs use the unchanged filters and\nexact fallback. Consumers must use signs, not assume the old estimate bits.')
source=source.replace('from warpower import','from . import').replace('from warpower._delaunay_certificate import symbolic_sphere','from ._delaunay_certificate import symbolic_sphere as legacy_symbolic_sphere').replace('from warpower._mesh_topology import face_vertices\n','')
source=source.replace('def symbolic(', 'def symbolic_sphere(').replace('return symbolic_sphere(points,ids,query)','return legacy_symbolic_sphere(points,ids,query)')
Path('warpower/_analytic_predicates.py').write_text(source)
replacement=reference.replace('fp.orient3d(', 'af.orient3d(').replace('result = symbolic_sphere(', 'result = af.symbolic_sphere(')
assert replacement!=reference
updated=old[:start]+replacement+old[end:]
updated=updated.replace('from . import _filtered_predicates as fp','from . import _analytic_predicates as af\nfrom . import _filtered_predicates as fp')
old_path.write_text(updated)
tests=(w/'test_analytic_filters.py').read_text().replace('from analytic_face_search import','from warpower._analytic_predicates import')
Path('tests/test_analytic_predicates.py').write_text(tests)
queries=(w/'test_analytic_queries.py').read_text()
queries=queries.replace('from warpower._proved_completion import search_unproved_faces','from warpower._proved_completion import search_unproved_faces as search_tree')
queries=queries.replace('from analytic_face_search import search_analytic as search_tree','from warpower import _filtered_predicates as fp\nfrom warpower._delaunay_certificate import symbolic_sphere\nfrom warpower._mesh_topology import face_vertices\n\nwp.set_module_options({"enable_backward":False,"fast_math":False,"fuse_fp":False})\n\n# Independent production baseline: the prior exact/ball search, without analytic filters.\n'+reference)
queries=queries.replace("[__name__,'analytic_face_search','warpower._facet_completion','warpower._proved_completion']","[__name__,'warpower._facet_completion','warpower._proved_completion']")
Path('tests/test_analytic_face_search.py').write_text(queries)
