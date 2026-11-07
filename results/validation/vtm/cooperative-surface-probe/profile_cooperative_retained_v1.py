"""Static retained-cell classifier comparisons; no complete-solve claim."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower import _selective
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
import cooperative_retained_v1 as kernels


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=Path('.work/vtm/cooperative-retained-profile-v1.json')
    assert not out.exists()
    report=dict(scope='Static current final-mesh ordinary retained classifier only. No whole-solve claim.',
                runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={p:sha(p) for p in [__file__,kernels.__file__,'.work/vtm/cooperative_rays_v2.py','warpower/_selective.py']},cases=[])
    wp.load_module(module=kernels,device='cuda:0',block_dim=32)
    wp.load_module(module=_selective,device='cuda:0',block_dim=32)
    for case in ['torus1000','torus4096','armadillo-uniform','armadillo-graded']:
        path=Path('results/validation/vtm/recovery-classification/recovery-classification-final')/(case+'-after')
        path=path/'mesh.npz' if path.is_dir() else path.with_suffix('.npz')
        stored=np.load(path)
        domain=warpower.SurfaceDomain(stored['input_points'],stored['input_faces'],surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
        e=warpower.ExtractionWorkspace(domain,stored['points'],stored['full_tetrahedra'])
        assert domain.error.numpy()[0]==0
        graphs={};arrays={}
        for name in ['reference','retained_1','retained_2','retained_4']:
            error=wp.zeros(1,dtype=int);keep=wp.zeros(e.capacity,dtype=int);quality=wp.zeros(e.capacity,dtype=wp.float64)
            args=[domain.tree,e.points,e.input_tetrahedra,e.input_count,e.tolerance,error,keep,quality]
            with wp.ScopedCapture() as cap:
                error.zero_()
                if name=='reference':
                    wp.launch(_selective.classify_retained,e.capacity,inputs=args,block_dim=32)
                else:
                    group=int(name[-1])
                    wp.launch_tiled(getattr(kernels,name),dim=(e.capacity+32//group-1)//(32//group),inputs=args,block_dim=32)
            graphs[name]=cap.graph;arrays[name]=(error,keep,quality)
            for _ in range(3):wp.capture_launch(cap.graph)
        wp.synchronize()
        initial=cpu_quota_snapshot();samples={k:[] for k in graphs}
        begin,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for repeat in range(20):
            order=list(graphs)
            if repeat%2:order.reverse()
            for name in order:
                wp.record_event(begin);wp.capture_launch(graphs[name]);wp.record_event(end)
                wp.synchronize_event(end);samples[name].append(wp.get_event_elapsed_time(begin,end))
        quota=cpu_quota_delta(initial,cpu_quota_snapshot())
        reference=[a.numpy() for a in arrays['reference']]
        values={k:[a.numpy() for a in aa] for k,aa in arrays.items()}
        row=dict(case=case,input=str(path),input_sha256=sha(path),tetrahedra=e.capacity,
                 samples_ms=samples,median_ms={k:float(np.median(v)) for k,v in samples.items()},
                 cpu_quota_delta=quota,quota_clean=all(v['throttled_periods']==0 for v in quota.values()),variants={})
        for name,v in values.items():
            row['variants'][name]=dict(error=int(v[0][0]),keep_mismatches=int(np.count_nonzero(v[1]!=reference[1])),
                                      quality_mismatches=int(np.count_nonzero(v[2]!=reference[2])),
                                      quality_max_abs_difference=float(np.max(np.abs(v[2]-reference[2]))),
                                      output_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in v])
        report['cases'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
        print(case,row['median_ms'],row['variants'],'clean',row['quota_clean'],flush=True)
    report['finished']=True
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
