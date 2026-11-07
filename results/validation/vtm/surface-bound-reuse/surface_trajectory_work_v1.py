"""Observe current full-extractor query work without replacing production kernels."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower import _extraction
from warpower._surface_bvh import SurfaceTree,closest_point,material_side
from surface_work_probe_v1 import counted_nearest,counted_side,same

wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.kernel
def observe_full(
    tree:SurfaceTree,centers:wp.array[wp.vec3d],radii:wp.array[wp.float64],
    count:wp.array[int],error:wp.array[int],distances:wp.array[wp.float64],keep:wp.array[int],
    tolerance:wp.float64,totals:wp.array2d[wp.int64],overview:wp.array[wp.int64],
):
    i=wp.tid()
    if i==0:
        wp.atomic_add(overview,0,wp.int64(1))
        wp.atomic_add(overview,1,wp.int64(error[0]!=0))
    if error[0]!=0 or i>=count[0]:
        return
    p=centers[i]
    hit=counted_nearest(tree,p,wp.float64(1.0e150))
    ref=closest_point(tree,p,wp.float64(1.0e150))
    wp.atomic_add(overview,2,wp.int64(1))
    wp.atomic_add(overview,3,wp.int64(1-same(hit.hit,ref)))
    for k in range(9):
        if k==7:wp.atomic_max(totals,0,k,wp.int64(hit.work[k]))
        else:wp.atomic_add(totals,0,k,wp.int64(hit.work[k]))
    side=int(1)
    if hit.hit.distance2>wp.float64(0.0):
        rays=counted_side(tree,p)
        side=rays.result[0]
        if rays.result[1]!=0:side=128
        wp.atomic_add(overview,4,wp.int64(side!=material_side(tree,p)))
        for k in range(9):wp.atomic_add(totals,1,k,wp.int64(rays.work[k]))
    distance=wp.sqrt(hit.hit.distance2)
    signed=wp.where(side==1,-distance,distance)
    retained=int(side==1 or distance/radii[i]<tolerance)
    wp.atomic_add(overview,5,wp.int64(signed!=distances[i]))
    wp.atomic_add(overview,6,wp.int64(retained!=keep[i]))


def canonical(a):
    a=np.sort(a,axis=1);return a[np.lexsort(a.T[::-1])]


def oriented(a):return sorted(min(tuple(f),tuple(np.roll(f,1)),tuple(np.roll(f,2))) for f in a)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    reference=json.loads(args.baseline.read_text());params=reference['parameters'];stored=np.load(args.mesh)
    totals=wp.zeros((2,9),dtype=wp.int64,device='cuda:0');overview=wp.zeros(7,dtype=wp.int64,device='cuda:0')
    wp.load_module(module=__name__,device='cuda:0',block_dim=32)
    original=_extraction.ExtractionWorkspace._classify_with
    def observe(self,kernel):
        original(self,kernel)
        if kernel is _extraction.classify:
            wp.launch(observe_full,self.capacity,inputs=[self.domain.tree,self.circumcenters,self.circumradii,self.input_count,self.error,self.signed_distance,self.selected,self.tolerance,totals,overview],device=self.device,block_dim=32)
    _extraction.ExtractionWorkspace._classify_with=observe
    inputs,faces=stored['input_points'],stored['input_faces']
    domain=warpower.SurfaceDomain(inputs,faces,surface_subdivisions=params['subdivisions'],crease_subdivisions=params.get('crease_subdivisions',4),feature_angle=params.get('feature_angle',45),corner_angle=params.get('corner_angle',45))
    sizing=None
    if params['case'].startswith('graded-'):
        lower,upper=inputs.min(axis=0),inputs.max(axis=0);anchor=(lower+upper)/2;anchor[0]=lower[0]
        sizing=warpower.SizingField(wp.array(anchor[None],dtype=wp.vec3),wp.array([.35],dtype=float),gradation=.65/float(upper[0]-lower[0]),minimum=.35,maximum=1)
    mesh=warpower.VariationalMeshingWorkspace(domain,params.get('capacity') or params['n'],target_count=params['n'],initial_count=params['initial'],seed=params['seed'],growth_iterations=params.get('growth_iterations',5),polish_iterations=params.get('polish_iterations',20),selective_iterations=params.get('selective_iterations',10),quality_rounds=params['quality_rounds'],automatic_sizing=params['automatic_sizing'],sizing=sizing)
    with wp.ScopedCapture() as cap:
        totals.zero_();overview.zero_();domain.refit()
        if sizing is not None:sizing.refit()
        mesh.run()
    rows=[]
    for replay in range(2):
        wp.capture_launch(cap.graph);wp.synchronize()
        assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
        counts=mesh.counts.numpy().tolist();assert counts==reference['counts']
        for attr,key in [('rounds','rounds'),('steps','steps' if 'steps' in reference else 'optimizer_steps'),('moves','moves' if 'moves' in reference else 'quality_moves')]:
            assert getattr(mesh,attr).numpy().tolist()==reference[key]
        for key in ['quality_rejections','refinement_rejections']:assert int(getattr(mesh,key).numpy()[0])==reference[key]
        np.testing.assert_array_equal(mesh.points.numpy()[:counts[0]].view(np.uint32),stored['points'].view(np.uint32))
        np.testing.assert_array_equal(mesh.core.boundary.roles.numpy()[:counts[0]],stored['roles'])
        for attr,key,count in [('tetrahedra','tetrahedra',counts[2]),('full_tetrahedra','full_tetrahedra',counts[1])]:
            np.testing.assert_array_equal(canonical(getattr(mesh,attr).numpy()[:count]),canonical(stored[key]))
        assert oriented(mesh.boundary_faces.numpy()[:counts[3]])==oriented(stored['boundary'])
        summary=overview.numpy().tolist();assert summary[1]==0 and all(x==0 for x in summary[3:])
        rows.append(dict(replay=replay,overview=summary,totals=totals.numpy().tolist(),points_bitwise_equal=True,topology_roles_work_equal=True))
    assert rows[0]['totals']==rows[1]['totals'] and rows[0]['overview']==rows[1]['overview']
    result=dict(scope='Two instrumented complete solves. Every original kernel executes; a read-only observer runs after full classification on actual current output centers. Excludes private recovered and ordinary retained classifiers and surface-summary queries. No instrumented timing claim.',
                runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'.work/vtm/surface_work_probe_v1.py','warpower/_surface_bvh.py']},
                baseline=str(args.baseline),mesh=str(args.mesh),mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                overview_columns=['full_calls','error_calls','queries','nearest_observer_mismatches','ray_observer_mismatches','actual_distance_mismatches','actual_keep_mismatches'],
                totals_columns=['query_or_direction_calls','visited_nodes','pruned_nodes','internal_nodes','leaves','triangle_evaluations','improvements_or_hits','max_stack_or_ambiguous_hits','box_evaluations'],replays=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
