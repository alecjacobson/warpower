"""Native production hierarchical exterior proof inside private uniform8K recovery; profiling only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import tangential_topology_recovery_v2 as experiment
experiment.install()
from examples.vtm_inputs import read_off
from warpower import SurfaceDomain

wp.set_module_options({'enable_backward':False})
@wp.kernel
def topology_begin(flags:wp.array[int]):flags[0]+=1
@wp.kernel
def topology_end(flags:wp.array[int]):flags[1]+=1
@wp.kernel
def quality_begin(flags:wp.array[int]):flags[2]+=1
@wp.kernel
def quality_end(flags:wp.array[int]):flags[3]+=1

Recovery=experiment.TopologyRecovery
class MarkedRecovery(Recovery):
    def __init__(self,owner,**kwargs):
        self.markers=owner.markers
        super().__init__(owner,**kwargs)
    def _record(self):
        wp.launch(topology_begin,1,inputs=[self.markers],device=self.device)
        super()._record()
        wp.launch(topology_end,1,inputs=[self.markers],device=self.device)
experiment.TopologyRecovery=MarkedRecovery
class MarkedMesher(experiment.Mesher):
    def __init__(self,domain,*args,**kwargs):
        self.markers=wp.zeros(4,dtype=int,device=domain.device)
        wp.load_module(module=__name__,device=domain.device)
        super().__init__(domain,*args,**kwargs)
    def _improve(self):
        wp.launch(quality_begin,1,inputs=[self.markers],device=self.device)
        super()._improve()
        wp.launch(quality_end,1,inputs=[self.markers],device=self.device)

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
inputs=read_off(Path('.work/vtm/scanned-armadillo-input/armadillo.off'))
domain=SurfaceDomain(*inputs,surface_subdivisions=1,crease_subdivisions=1,feature_angle=120,corner_angle=45)
mesh=MarkedMesher(domain,8192,initial_count=256,seed=2026,quality_rounds=4)
with wp.ScopedCapture() as cap:domain.refit();mesh.run()
wp.capture_launch(cap.graph);wp.synchronize();assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
wp.capture_debug_dot_print(cap.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart();wp.capture_launch(cap.graph);wp.synchronize();cuda.cudaProfilerStop()
assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
n,full,m,b=mesh.counts.numpy().tolist()
arrays=dict(points=mesh.points.numpy()[:n],full_tetrahedra=mesh.full_tetrahedra.numpy()[:full],tetrahedra=mesh.tetrahedra.numpy()[:m],boundary=mesh.boundary_faces.numpy()[:b],roles=mesh.core.boundary.roles.numpy()[:n],radius_ratio=mesh.radius_ratio.numpy()[:m],input_points=inputs[0],input_faces=inputs[1])
np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
r=dict(scope=__doc__,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/tangential_topology_recovery_v2.py'),Path('warpower/_exterior_proof_bvh.py'),Path('warpower/_proven_exterior.py')]},markers=mesh.markers.numpy().tolist(),work={k:getattr(mesh,k).numpy().tolist() for k in ['counts','rounds','steps','moves','topology','error','valid']},recovery={k:getattr(mesh.topology_recovery,k).numpy().tolist() for k in ['chosen','accepted','attempts','invoked']})
def canonical(t):
 s=np.sort(t,axis=1);return s[np.lexsort(s.T[::-1])]
ref=Path('.work/vtm/tangential-topology-pilot-v2/uniform8192')
with np.load(ref/'mesh.npz') as d:
 r['points_bitwise_equal']=bool(np.array_equal(d['points'].view(np.uint32),arrays['points'].view(np.uint32)))
 r['roles_equal']=bool(np.array_equal(d['roles'],arrays['roles']))
 r['canonical_cells_equal']={k:bool(np.array_equal(canonical(d[k]),canonical(arrays[k]))) for k in ['full_tetrahedra','tetrahedra','boundary']}
d=json.loads((ref/'result.json').read_text());r['work_equal']={k:r['work'][k]==d[k] for k in ['counts','rounds','steps','moves','topology']}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
assert r['markers']==[2,2,2,2] and all(r['work_equal'].values()) and all(r['canonical_cells_equal'].values()) and r['roles_equal']
