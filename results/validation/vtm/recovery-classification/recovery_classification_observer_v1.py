"""#194 observation only: all original classification work retained."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower import _extraction,_mesher,_quality_moves,_selective

from warpower._surface_bvh import SurfaceTree, closest_point, closest_triangle, material_side
from warpower._wide_center import exact_geometry, uncertain
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def classify_observed(
    tree: SurfaceTree,
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tolerance: wp.float64,
    error: wp.array[int],
    centers: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    keep: wp.array[int],
    volumes: wp.array[wp.float64],
    quality: wp.array[wp.float64],
    queries: wp.array[wp.int64],
):
    i = wp.tid()
    keep[i] = 0
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    for k in range(4):
        if ids[k] < 0 or ids[k] >= points.shape[0]:
            wp.atomic_or(error, 0, 4)
            return
        p = points[ids[k]]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            wp.atomic_or(error, 0, 4)
            return
    p0 = wp.vec3d(points[ids[0]])
    a, b, c = (
        wp.vec3d(points[ids[1]]) - p0,
        wp.vec3d(points[ids[2]]) - p0,
        wp.vec3d(points[ids[3]]) - p0,
    )
    bc, ca, ab = wp.cross(b, c), wp.cross(c, a), wp.cross(a, b)
    determinant = wp.dot(a, bc)
    area2 = wp.length(bc) + wp.length(ca) + wp.length(ab) + wp.length(wp.cross(b - a, c - a))
    center, radius = wp.vec3d(), wp.float64(0.0)
    if uncertain(a, b, c, determinant):
        geometry = exact_geometry(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]])
        if geometry.valid == 0:
            wp.atomic_or(error, 0, 8)
            return
        determinant = wp.float64(geometry.determinant)
        center = wp.vec3d(geometry.center)
        radius = wp.float64(geometry.radius)
        area2 = wp.float64(geometry.area2)
    else:
        if determinant <= wp.float64(0.0):
            wp.atomic_or(error, 0, 8)
            return
        offset = (wp.dot(a, a) * bc + wp.dot(b, b) * ca + wp.dot(c, c) * ab) / (
            wp.float64(2.0) * determinant
        )
        center, radius = p0 + offset, wp.length(offset)
    if not wp.isfinite(radius) or radius <= wp.float64(0.0):
        wp.atomic_or(error, 0, 8)
        return
    wp.atomic_add(queries, 0, wp.int64(1))
    bounded = wp.abs(center[0]) < wp.float64(1.0e140) and wp.abs(center[1]) < wp.float64(1.0e140) and wp.abs(center[2]) < wp.float64(1.0e140)
    proved = False
    if bounded:
        wp.atomic_add(queries, 1, wp.int64(1))
        if tree.n > 0:
            f = tree.ids[0]
            x = wp.vec3d(tree.points[tree.indices[3*f]])
            y = wp.vec3d(tree.points[tree.indices[3*f+1]])
            z = wp.vec3d(tree.points[tree.indices[3*f+2]])
            q = closest_triangle(center,x,y,z)
            d = wp.length_sq(center-q)
            limit = wp.float64(1.0e150)
            proved = wp.isfinite(d) and d < limit*limit
    if proved:
        wp.atomic_add(queries, 2, wp.int64(1))
    hit = closest_point(tree, center, wp.float64(1.0e150))
    if hit.face < 0 or not wp.isfinite(hit.distance2):
        wp.atomic_add(queries, 3, wp.int64(1))
        wp.atomic_or(error, 0, 16)
        return
    side = int(1)
    ray_side = int(1)
    if hit.distance2 > wp.float64(0.0):
        side = material_side(tree, center)
        ray_side = side
    else:
        wp.atomic_add(queries, 4, wp.int64(1))
        if proved:
            ray_side = material_side(tree, center)
    if side == 1:
        wp.atomic_add(queries, 5, wp.int64(1))
    elif side == 0:
        wp.atomic_add(queries, 6, wp.int64(1))
    else:
        wp.atomic_add(queries, 7, wp.int64(1))
    if proved and ray_side == 1:
        wp.atomic_add(queries, 8, wp.int64(1))
    if side != 0 and side != 1:
        wp.atomic_or(error, 0, 16)
        return
    distance = wp.sqrt(hit.distance2)
    centers[i] = center
    radii[i] = radius
    distances[i] = wp.where(side == 1, -distance, distance)
    keep[i] = int(side == 1 or distance / radius < tolerance)
    volumes[i] = determinant / wp.float64(6.0)
    quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))

labels=['other','current_selection','selection_guard','quality_initial','quality_trial','quality_restore','mesher_extract']
category=0
query_columns=['finite_geometry_queries','bounded_centers','finite_witnesses','failed_full_queries','exact_zero_distances','inside_including_zero','outside','ambiguous','eligible_inside_shortcuts']

@wp.kernel
def record_use(kind:int,inputs:wp.array[int],cells:wp.array[int],faces:wp.array[int],error:wp.array[int],valid:wp.array[int],counters:wp.array2d[wp.int64]):
    wp.atomic_add(counters,kind,0,wp.int64(1))
    wp.atomic_add(counters,kind,1,wp.int64(inputs[0]))
    wp.atomic_add(counters,kind,2,wp.int64(cells[0]))
    wp.atomic_add(counters,kind,3,wp.int64(faces[0]))
    wp.atomic_add(counters,kind,4,wp.int64(error[0]!=0))
    wp.atomic_add(counters,kind,5,wp.int64(valid[0]==0))


def install():
    counters=wp.zeros((len(labels),6),dtype=wp.int64,device='cuda:0')
    queries=wp.zeros(len(query_columns),dtype=wp.int64,device='cuda:0')
    wp.load_module(module=__name__,device='cuda:0')
    launch=_extraction.ExtractionWorkspace._launch
    def observed_launch(self,kernel,dim,args,*,block_dim=256):
        if kernel is _extraction.classify and isinstance(self,_extraction._SelectionExtractionWorkspace):
            kernel=classify_observed;args=list(args)+[queries]
        return launch(self,kernel,dim,args,block_dim=block_dim)
    _extraction.ExtractionWorkspace._launch=observed_launch
    def wrap(cls,name,kind):
        original=getattr(cls,name)
        def context(self,*args,**kwargs):
            global category
            previous=category;category=kind
            try:return original(self,*args,**kwargs)
            finally:category=previous
        setattr(cls,name,context)
    for cls,name,kind in [(_selective.SelectiveUpdate,'_select',1),(_selective.SelectiveUpdate,'_check',2),(_quality_moves.QualityMove,'_initial',3),(_quality_moves.QualityMove,'_complete_quality',4),(_quality_moves.QualityMove,'_restore',5),(_mesher.VariationalMeshingWorkspace,'_extract',6)]:wrap(cls,name,kind)
    original=_extraction.ExtractionWorkspace._record
    def observe(self):
        kind=category
        original(self)
        wp.launch(record_use,1,inputs=[kind,self.input_count,self.tetrahedra_count,self.boundary_count,self.error,self.valid,counters],device=self.device)
    _extraction.ExtractionWorkspace._record=observe
    return counters,queries


def canonical(a):
    rows=np.sort(a,axis=1);return rows[np.lexsort(tuple(rows[:,j] for j in reversed(range(rows.shape[1]))))]

def oriented(a):
    rows=np.asarray([tuple(np.roll(row,-np.argmin(row))) for row in a]);return rows[np.lexsort((rows[:,2],rows[:,1],rows[:,0]))]

p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
reference=json.loads(args.baseline.read_text());params=reference['parameters'];stored=np.load(args.mesh);inputs=stored['input_points'];faces=stored['input_faces']
counters,queries=install()
domain=warpower.SurfaceDomain(inputs,faces,surface_subdivisions=params['subdivisions'],crease_subdivisions=params.get('crease_subdivisions',4),feature_angle=params.get('feature_angle',45),corner_angle=params.get('corner_angle',45))
sizing=None
if params['case'].startswith('graded-'):
    lower,upper=inputs.min(axis=0),inputs.max(axis=0);anchor=(lower+upper)/2;anchor[0]=lower[0]
    sizing=warpower.SizingField(wp.array(anchor[None],dtype=wp.vec3,device=domain.device),wp.array([0.35],dtype=float,device=domain.device),gradation=0.65/float(upper[0]-lower[0]),minimum=0.35,maximum=1)
mesh=warpower.VariationalMeshingWorkspace(domain,params.get('capacity') or params['n'],target_count=params['n'],initial_count=params['initial'],seed=params['seed'],growth_iterations=params.get('growth_iterations',5),polish_iterations=params.get('polish_iterations',20),selective_iterations=params.get('selective_iterations',10),quality_rounds=params['quality_rounds'],automatic_sizing=params['automatic_sizing'],sizing=sizing)
with wp.ScopedCapture() as capture:
    counters.zero_();queries.zero_();domain.refit()
    if sizing is not None:sizing.refit()
    mesh.run()
rows=[]
for replay in range(2):
    wp.capture_launch(capture.graph)
    assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
    counts=mesh.counts.numpy().tolist();assert counts==reference['counts']
    for attr,key in [('rounds','rounds'),('steps','steps' if 'steps' in reference else 'optimizer_steps'),('moves','moves' if 'moves' in reference else 'quality_moves')]:assert getattr(mesh,attr).numpy().tolist()==reference[key],(attr,key)
    for key in ['quality_rejections','refinement_rejections']:assert int(getattr(mesh,key).numpy()[0])==reference[key]
    values=dict(points=mesh.points.numpy()[:counts[0]],tetrahedra=mesh.tetrahedra.numpy()[:counts[2]],full_tetrahedra=mesh.full_tetrahedra.numpy()[:counts[1]],boundary=mesh.boundary_faces.numpy()[:counts[3]],roles=mesh.core.boundary.roles.numpy()[:counts[0]],input_points=inputs,input_faces=faces)
    for name in ['tetrahedra','full_tetrahedra']:np.testing.assert_array_equal(canonical(values[name]),canonical(stored[name]))
    np.testing.assert_array_equal(oriented(values['boundary']),oriented(stored['boundary']));np.testing.assert_array_equal(values['roles'],stored['roles'])
    delta=np.argwhere(values['points'].view(np.uint32)!=stored['points'].view(np.uint32))
    comparison=dict(points_bitwise_equal=len(delta)==0,maximum_coordinate_difference=float(np.max(np.abs(values['points']-stored['points']))),changed_coordinate_values=[[int(i),int(j),float(stored['points'][i,j]),float(values['points'][i,j])] for i,j in delta],full_and_material_topology_equal=True,oriented_boundary_equal=True,roles_equal=True,work_equal=True)
    archive=None
    if len(delta):
        archive=args.output.with_name(args.output.stem+'-replay'+str(replay)+'.npz');np.savez_compressed(archive,**values)
    raw=counters.numpy();assert raw[0,0]==0
    rows.append(dict(replay=replay,queries=dict(zip(query_columns,queries.numpy().tolist())),counters={name:raw[i].tolist() for i,name in enumerate(labels)},comparison=comparison,changed_mesh=str(archive) if archive else None))
assert rows[0]['counters']==rows[1]['counters']
assert rows[0]['queries']==rows[1]['queries']
report=dict(scope='Observation only; every original classification nearest/ray query executes. Only private selection classification collects query categories; public full distances remain unchanged. Merged baseline includes the separately validated private surface-status path. Caller context is fixed while recording; counters execute on device. No performance claim. Two complete fresh solves compare against the merged production result. Bitwise-equal points plus canonical topology/roles/boundaries refer to the existing archive; changed point arrays are retained separately for independent validation.',source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline=str(args.baseline),mesh_reference=str(args.mesh),mesh_reference_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),columns=['calls','sum_input_cells','sum_output_cells','sum_boundary_faces','error_calls','invalid_calls'],replays=rows)
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
