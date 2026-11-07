"""#192 observation only: classify extraction callers without omitting any work."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower import _extraction,_mesher,_quality_moves,_selective

wp.set_module_options({'enable_backward':False})
labels=['other','current_selection','selection_guard','quality_initial','quality_trial','quality_restore','mesher_extract']
category=0

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
    wp.load_module(module=__name__,device='cuda:0')
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
    return counters


def canonical(a):
    rows=np.sort(a,axis=1);return rows[np.lexsort(tuple(rows[:,j] for j in reversed(range(rows.shape[1]))))]

def oriented(a):
    rows=np.asarray([tuple(np.roll(row,-np.argmin(row))) for row in a]);return rows[np.lexsort((rows[:,2],rows[:,1],rows[:,0]))]

p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
reference=json.loads(args.baseline.read_text());params=reference['parameters'];stored=np.load(args.mesh);inputs=stored['input_points'];faces=stored['input_faces']
counters=install()
domain=warpower.SurfaceDomain(inputs,faces,surface_subdivisions=params['subdivisions'],crease_subdivisions=params.get('crease_subdivisions',4),feature_angle=params.get('feature_angle',45),corner_angle=params.get('corner_angle',45))
sizing=None
if params['case'].startswith('graded-'):
    lower,upper=inputs.min(axis=0),inputs.max(axis=0);anchor=(lower+upper)/2;anchor[0]=lower[0]
    sizing=warpower.SizingField(wp.array(anchor[None],dtype=wp.vec3,device=domain.device),wp.array([0.35],dtype=float,device=domain.device),gradation=0.65/float(upper[0]-lower[0]),minimum=0.35,maximum=1)
mesh=warpower.VariationalMeshingWorkspace(domain,params.get('capacity') or params['n'],target_count=params['n'],initial_count=params['initial'],seed=params['seed'],growth_iterations=params.get('growth_iterations',5),polish_iterations=params.get('polish_iterations',20),selective_iterations=params.get('selective_iterations',10),quality_rounds=params['quality_rounds'],automatic_sizing=params['automatic_sizing'],sizing=sizing)
with wp.ScopedCapture() as capture:
    counters.zero_();domain.refit()
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
    rows.append(dict(replay=replay,counters={name:raw[i].tolist() for i,name in enumerate(labels)},comparison=comparison,changed_mesh=str(archive) if archive else None))
assert rows[0]['counters']==rows[1]['counters']
report=dict(scope='Observation only; every full extraction/diagnostic still executes. Caller context is fixed while recording; counters execute on device. No performance claim. Two complete fresh solves compare against the merged production result. Bitwise-equal points plus canonical topology/roles/boundaries refer to the existing archive; changed point arrays are retained separately for independent validation.',source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline=str(args.baseline),mesh_reference=str(args.mesh),mesh_reference_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),columns=['calls','sum_input_cells','sum_output_cells','sum_boundary_faces','error_calls','invalid_calls'],replays=rows)
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
