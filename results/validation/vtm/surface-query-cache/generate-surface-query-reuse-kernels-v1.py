"""Retain original arithmetic/traversal and append observations of actual query inputs/results."""
import ast,hashlib,json
from pathlib import Path
w=Path('.work/vtm');target=w/'surface_query_reuse_kernels_v1.py';assert not target.exists()
head='''import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point, material_side
from warpower._wide_center import exact_geometry, uncertain
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.struct
class QueryState:
    centers:wp.array[wp.vec3d]
    valid:wp.array[int]
    rays:wp.array[int]
    sides:wp.array[int]
    nearest:wp.array[int]
    limits:wp.array[wp.float64]
    distances:wp.array[wp.float64]
    faces:wp.array[int]
    hit_points:wp.array[wp.vec3d]

'''
sources={}
for filename,name,newname in [('warpower/_extraction.py','classify','observe_full'),('warpower/_selective.py','classify_retained','observe_retained')]:
    p=Path(filename);text=p.read_text();sources[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    s='@wp.kernel\n'+ast.get_source_segment(text,node)
    s=s.replace('def '+name+'(','def '+newname+'(',1)
    s=s.replace('):\n    i = wp.tid()','    probe: QueryState,\n):\n    i = wp.tid()',1)
    s=s.replace('    keep[i] = 0','    keep[i] = 0\n    probe.valid[i] = 0\n    probe.rays[i] = 0\n    probe.nearest[i] = 0',1)
    if name=='classify':
        old='    hit = closest_point(tree, center, wp.float64(1.0e150))'
        s=s.replace(old,'    probe.centers[i] = center\n    probe.valid[i] = 1\n    probe.nearest[i] = 1\n    probe.limits[i] = wp.float64(1.0e150)\n'+old+'\n    probe.distances[i] = hit.distance2\n    probe.faces[i] = hit.face\n    probe.hit_points[i] = hit.point')
        s=s.replace('        side = material_side(tree, center)','        probe.rays[i] = 1\n        side = material_side(tree, center)\n        probe.sides[i] = side')
    else:
        old='    side = material_side(tree, center)'
        s=s.replace(old,'    probe.centers[i] = center\n    probe.valid[i] = 1\n    probe.rays[i] = 1\n'+old+'\n    probe.sides[i] = side')
        old='        hit = closest_point(tree, center, limit)'
        s=s.replace(old,'        probe.nearest[i] = 1\n        probe.limits[i] = limit\n'+old+'\n        probe.distances[i] = hit.distance2\n        probe.faces[i] = hit.face\n        probe.hit_points[i] = hit.point')
    assert s.count('probe.centers[i] = center')==1 and s.count('probe.hit_points[i] = hit.point')==1
    head+=s+'\n\n'
target.write_text(head)
(w/'surface-query-reuse-generated-v1.json').write_text(json.dumps(dict(scope=__doc__,original_source_sha256=sources,generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
