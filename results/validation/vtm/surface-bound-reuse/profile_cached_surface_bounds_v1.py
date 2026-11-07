"""Static current final-center full-nearest query comparison, no solve claim."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower._surface_bvh import Closest
from extraction_split import prepare_geometry
from benchmarks.run import cpu_quota_snapshot,cpu_quota_delta
import cached_surface_bounds_v1 as kernels

out=Path('.work/vtm/cached-surface-bounds-profile-v1.json');assert not out.exists()
report=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            scope='Static full-nearest query kernels; no geometry preparation/setup or whole-solve speed claim.',
            sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,kernels.__file__,'.work/vtm/extraction_split.py','warpower/_surface_bvh.py']},cases=[])
wp.load_module(module=kernels,device='cuda:0',block_dim=32)
wp.load_module(module='extraction_split',device='cuda:0',block_dim=32)
for case in ['torus1000','torus4096','armadillo-uniform','armadillo-graded']:
    path=Path('results/validation/vtm/recovery-classification/recovery-classification-final')/(case+'-after')
    path=path/'mesh.npz' if path.is_dir() else path.with_suffix('.npz')
    d=np.load(path);domain=warpower.SurfaceDomain(d['input_points'],d['input_faces'],surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
    e=warpower.ExtractionWorkspace(domain,d['points'],d['full_tetrahedra']);e.error.zero_()
    args=[domain.tree,e.points,e.input_tetrahedra,e.input_count,e.tolerance,e.error,e.circumcenters,e.circumradii,e.signed_distance,e.selected,e._input_volume,e._input_quality]
    wp.launch(prepare_geometry,e.capacity,inputs=args,block_dim=32);assert e.error.numpy()[0]==domain.error.numpy()[0]==0
    limits=wp.full(e.capacity,1e150,dtype=wp.float64)
    arrays={k:wp.empty(e.capacity,dtype=Closest) for k in ['reference','candidate']};graphs={}
    for name in arrays:
        with wp.ScopedCapture() as cap:wp.launch(getattr(kernels,name),e.capacity,inputs=[domain.tree,e.circumcenters,limits,arrays[name]],block_dim=32)
        graphs[name]=cap.graph
        for _ in range(3):wp.capture_launch(cap.graph)
    wp.synchronize();initial=cpu_quota_snapshot();samples={k:[] for k in arrays}
    begin,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
    for repeat in range(20):
        order=list(arrays)
        if repeat%2:order.reverse()
        for name in order:
            wp.record_event(begin);wp.capture_launch(graphs[name]);wp.record_event(end);wp.synchronize_event(end)
            samples[name].append(wp.get_event_elapsed_time(begin,end))
    quota=cpu_quota_delta(initial,cpu_quota_snapshot())
    a,b=[x.numpy() for x in arrays.values()]
    differences={field:int(np.count_nonzero(a[field]!=b[field])) for field in ['point','distance2','face']}
    row=dict(case=case,input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),queries=e.capacity,
             samples_ms=samples,median_ms={k:float(np.median(v)) for k,v in samples.items()},
             cpu_quota_delta=quota,quota_clean=all(v['throttled_periods']==0 for v in quota.values()),differences=differences,
             output_sha256={k:{field:hashlib.sha256(value.numpy()[field].tobytes()).hexdigest() for field in differences} for k,value in arrays.items()})
    report['cases'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
    print(case,row['median_ms'],differences,'clean',row['quota_clean'],flush=True)
    assert all(v==0 for v in differences.values())
report['finished']=True;out.write_text(json.dumps(report,indent=2)+'\n')
