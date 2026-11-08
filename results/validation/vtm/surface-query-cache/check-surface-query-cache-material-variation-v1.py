"""Exact local boundary-displacement bound and honest quality comparison for issue220."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np

w=Path('.work/vtm');out=w/'surface-query-cache-final'
dest=w/'surface-query-cache-material-variation-v1.json';assert not dest.exists()
report=json.loads((w/'surface-query-cache-final-independent-v2.json').read_text())
case=report['cases']['hand-seed1'];equivalence=report['equivalence']['hand-seed1']
assert report['geometry_complete'] and report['equivalence_failures']==['hand-seed1']
paths={phase:out/f'hand-seed1-{phase}/mesh.npz' for phase in ['before','after']}
data={phase:np.load(path) for phase,path in paths.items()}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(row):return tuple(sorted(map(int,row)))
def point(row):return tuple(F(float(x)) for x in row)
def det(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def volume6(points,t):
    a,b,c,d=[points[i] for i in t];return det(sub(b,a),sub(c,a),sub(d,a))
def edges(triangles):
    result=Counter()
    for t in triangles:
        for i in range(3):
            a,b=t[i],t[(i+1)%3];result[min(a,b),max(a,b)]+=1 if a<b else -1
    return {k:v for k,v in result.items() if v}
points={phase:[point(p) for p in d['points']] for phase,d in data.items()}
changes={}
for kind in ['full_tetrahedra','tetrahedra','boundary']:
    sets={phase:{key(t):tuple(map(int,t)) for t in d[kind]} for phase,d in data.items()}
    changes[kind]={phase:[sets[phase][t] for t in sorted(sets[phase].keys()-sets[other].keys())]
                   for phase,other in [('before','after'),('after','before')]}
assert all(len(x)==2 for x in changes['boundary'].values())
boundary_edges={phase:edges(ts) for phase,ts in changes['boundary'].items()}
assert boundary_edges['before']==boundary_edges['after'] and len(boundary_edges['before'])==4
assert all(abs(x)==1 for x in boundary_edges['before'].values())
quad=sorted({i for t in changes['boundary']['before'] for i in t});assert len(quad)==4
assert quad==[42,90,280,430]
def diagonal(ts):return sorted(set(ts[0])&set(ts[1]))
diagonals={phase:diagonal(ts) for phase,ts in changes['boundary'].items()}
# On the before frame both triangulations are graphs over the convex x,z quad.
# Their difference is piecewise affine, zero on its boundary, and reaches its
# maximum absolute value at the intersection of the two diagonals. Projecting
# vertically supplies a Euclidean distance upper bound in both directions.
cycle=[42,90,280,430]
p=points['before']
cross2=lambda a,b:a[0]*b[1]-a[1]*b[0]
project=lambda i:(p[i][0],p[i][2])
q=[project(i) for i in cycle]
turns=[cross2(sub(q[(i+1)%4],q[i]),sub(q[(i+2)%4],q[(i+1)%4])) for i in range(4)]
assert all(t>0 for t in turns) or all(t<0 for t in turns)
a,b=[project(i) for i in diagonals['before']]
c,d=[project(i) for i in diagonals['after']]
u,v=sub(b,a),sub(d,c);den=cross2(u,v);assert den
t=cross2(sub(c,a),v)/den;s=cross2(sub(c,a),u)/den
assert 0<t<1 and 0<s<1
y0=(1-t)*p[diagonals['before'][0]][1]+t*p[diagonals['before'][1]][1]
y1=(1-s)*p[diagonals['after'][0]][1]+s*p[diagonals['after'][1]][1]
graph_bound=abs(y0-y1)
# Every changed vertex moves along just one coordinate, so its exact Euclidean
# displacement is that absolute rational coordinate difference. Barycentric
# interpolation bounds movement of every point on a corresponding triangle.
vertex_bound=F(0);changed_coordinates=[]
for i,(a,b) in enumerate(zip(points['before'],points['after'])):
    diffs=[(j,y-x) for j,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert len(diffs)<=1
    for j,delta in diffs:
        vertex_bound=max(vertex_bound,abs(delta))
        changed_coordinates.append(dict(site=i,axis=j,before=float(a[j]),after=float(b[j]),exact_delta=str(delta)))
bound=vertex_bound+graph_bound
exact_volume={phase:sum(det(p[t[0]],p[t[1]],p[t[2]]) for t in data[phase]['boundary'])/6 for phase,p in points.items()}
assert all(v>0 for v in exact_volume.values())
cells=[]
for phase in ['before','after']:
    for kind in ['full_tetrahedra','tetrahedra']:
        for tet in changes[kind][phase]:
            v=volume6(points[phase],tet)/6;assert v>0
            cells.append(dict(phase=phase,kind=kind,vertices=tet,exact_volume=str(v),volume=float(v)))
for phase in ['before','after']:
    c=case[phase]['independent']
    assert c['mesh_sha256']==sha(paths[phase]) and c['full_hull']['valid']
    assert c['all_boundary_vertex_links_single_cycles'] and c['vtu_round_trip_exact'] and c['boundary_self_intersection_pairs']==0
assert case['before']['features']==case['after']['features']
assert all(equivalence['work_equal'].values())
quality={phase:case[phase]['independent']['quality'] for phase in ['before','after']}
metrics=['volume','relative_volume_error','mean_radius_ratio','radius_ratio_percentiles','dihedral_degrees_min','dihedral_degrees_max','two_sided_area_weighted_rms']
common={key(t):float(q) for t,q in zip(data['before']['tetrahedra'],data['before']['radius_ratio'])}
after={key(t):float(q) for t,q in zip(data['after']['tetrahedra'],data['after']['radius_ratio'])}
shared=common.keys()&after.keys()
r=dict(scope=__doc__,source_sha256=sha(Path(__file__)),independent_report_sha256=sha(w/'surface-query-cache-final-independent-v2.json'),
       mesh_sha256={phase:sha(path) for phase,path in paths.items()},changed_coordinates=changed_coordinates,
       changed_cells_and_boundary=changes,changed_cell_exact_volumes=cells,
       boundary_quad=quad,diagonals=diagonals,oriented_patch_boundary_edges_equal=True,
       projected_quad_strictly_convex=True,diagonal_intersection_parameters=[str(t),str(s)],
       same_frame_vertical_graph_bound_exact=str(graph_bound),same_frame_vertical_graph_bound=float(graph_bound),
       maximum_vertex_movement_exact=str(vertex_bound),two_sided_boundary_hausdorff_upper_bound_exact=str(bound),two_sided_boundary_hausdorff_upper_bound=float(bound),
       boundary_bound_proof='Changed patch triangulations cover the same convex projected quadrilateral in the before frame. Their graph difference has its only possible interior extremum at the diagonal intersection. All other corresponding faces move by at most the maximum vertex displacement, as does the after patch moved onto the before frame. Triangle inequality gives the recorded global two-sided upper bound. This is specific to these saved boundary meshes, not an input-to-output Hausdorff guarantee.',
       exact_material_volume={phase:str(v) for phase,v in exact_volume.items()},exact_material_volume_difference=str(exact_volume['after']-exact_volume['before']),
       material_volume_difference=float(exact_volume['after']-exact_volume['before']),
       quality={phase:{k:quality[phase][k] for k in metrics} for phase in ['before','after']},
       common_material_cells=len(shared),maximum_common_cell_radius_ratio_difference=max(abs(common[k]-after[k]) for k in shared),
       changed_material_quality={phase:[dict(vertices=t,radius_ratio=(common if phase=='before' else after)[key(t)]) for t in changes['tetrahedra'][phase]] for phase in ['before','after']},
       features_exactly_equal=True,recorded_work_equal=True,material_connectivity_equal=False,
       interpretation='Both own-frame meshes independently validate. Material connectivity and mean quality differ; this does not establish bitwise or quality identity, nor the cause of coordinate variation. Paired-query and repeat-run investigation remains required.')
dest.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['two_sided_boundary_hausdorff_upper_bound','material_volume_difference','common_material_cells','maximum_common_cell_radius_ratio_difference']},indent=2))
