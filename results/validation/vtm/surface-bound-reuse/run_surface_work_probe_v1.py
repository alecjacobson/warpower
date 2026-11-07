"""Record query operation counts, never timings of the instrumented queries."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from extraction_split import prepare_geometry
import surface_work_probe_v1 as kernels

root=Path.cwd();out=root/'.work/vtm/surface-work-probe-v1';assert not out.exists();out.mkdir()
report=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            scope='Instrumented query work on final snapshots, not elapsed-time attribution or intermediate trajectories.',
            source_sha256={str(Path(p).relative_to(root)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(kernels.__file__).resolve(),root/'warpower/_surface_bvh.py',root/'.work/vtm/extraction_split.py']},
            fields=dict(nearest=['calls','popped_nodes','pruned_nodes','internal_nodes','leaves','triangle_evaluations','improvements','max_stack','box_evaluations'],
                        rays=['direction_calls','visited_nodes','pruned_nodes','internal_nodes','leaves','triangle_evaluations','intersection_hits','ambiguous_hits','box_evaluations']),cases={})
wp.load_module(module=kernels,device='cuda:0',block_dim=32)
wp.load_module(module='extraction_split',device='cuda:0',block_dim=32)


def summarize(a,names):
    return {name:dict(total=int(a[:,i].astype(np.int64).sum()),mean=float(a[:,i].mean()),
                     percentiles=dict(zip(['0','50','90','99','100'],np.percentile(a[:,i],[0,50,90,99,100]).tolist()))) for i,name in enumerate(names)}


for case in ['torus1000','torus4096','armadillo-uniform','armadillo-graded']:
    path=root/'results/validation/vtm/recovery-classification/recovery-classification-final'/(case+'-after')
    path=path/'mesh.npz' if path.is_dir() else path.with_suffix('.npz')
    stored=np.load(path)
    domain=warpower.SurfaceDomain(stored['input_points'],stored['input_faces'],surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
    e=warpower.ExtractionWorkspace(domain,stored['points'],stored['full_tetrahedra'])
    assert domain.error.numpy()[0]==0
    e.error.zero_()
    args=[domain.tree,e.points,e.input_tetrahedra,e.input_count,e.tolerance,e.error,e.circumcenters,e.circumradii,e.signed_distance,e.selected,e._input_volume,e._input_quality]
    wp.launch(prepare_geometry,e.capacity,inputs=args,block_dim=32)
    assert e.error.numpy()[0]==0
    full,retained,rays=[wp.zeros(e.capacity,dtype=kernels.Counts) for _ in range(3)]
    status=wp.zeros((e.capacity,7),dtype=int)
    with wp.ScopedCapture() as cap:
        wp.launch(kernels.observe,e.capacity,inputs=[domain.tree,e.circumcenters,e.circumradii,e.input_count,e.error,e.tolerance,full,retained,rays,status],block_dim=32)
    wp.capture_launch(cap.graph);wp.synchronize()
    data={k:v.numpy() for k,v in dict(full=full,retained=retained,rays=rays,status=status).items()}
    archive=out/(case+'.npz');np.savez_compressed(archive,**data)
    states=data['status'];assert np.count_nonzero(states[:,[0,2,5]])==0
    assert np.all(states[:,1]==1)
    row=dict(input=str(path.relative_to(root)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),queries=e.capacity,faces=domain.tree.n,
             observer_mismatches=0,full_hit_successes=int(states[:,1].sum()),
             sides={str(k):int(v) for k,v in zip(*np.unique(states[:,3],return_counts=True))},
             retained_nearest_modes={str(k):int(v) for k,v in zip(*np.unique(states[:,4],return_counts=True))},
             retained_hits=int(states[:,6].sum()),
             full=summarize(data['full'],report['fields']['nearest']),
             retained=summarize(data['retained'],report['fields']['nearest']),
             rays=summarize(data['rays'],report['fields']['rays']))
    report['cases'][case]=row
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(case,{k:{f:row[k][f]['mean'] for f in ['box_evaluations','leaves','triangle_evaluations']} for k in ['full','retained','rays']},flush=True)
report['finished']=True
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
