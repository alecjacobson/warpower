"""Static query timing only; includes no complete VTM or adoption claim."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
import warpower
import cooperative_rays_v2 as kernels
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from extraction_split import prepare_geometry
from tests.test_domain import torus


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('.work/vtm/cooperative-rays-profile-v3.json')
    assert not out.exists()
    report = dict(scope='Warmed static surface queries. Unchanged prepared geometry; no preparation/full-solve speed claim.',
                  runtime=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  sources={p:sha(p) for p in [__file__,kernels.__file__,'.work/vtm/extraction_split.py','warpower/_surface_bvh.py']},
                  warp=wp.__version__,cases=[])
    wp.set_module_options({'fuse_fp':True}, module=kernels)
    wp.load_module(module=kernels,device='cuda:0',block_dim=32)
    wp.load_module(module='extraction_split',device='cuda:0',block_dim=32)
    names=['leaf_1','leaf_2','leaf_4','subtree_2','subtree_4','subtree_8','subtree_16','subtree_32']
    for case in ['torus1000','torus4096','armadillo-uniform','armadillo-graded','torus2304-random','torus9216-random','armadillo-small64']:
        owner=None
        if '-random' in case:
            p,f=torus(96,48) if case.startswith('torus9216') else torus()
            domain=warpower.SurfaceDomain(p,f,surface_subdivisions=1,crease_subdivisions=1)
            lo,hi=p.min(axis=0).astype(float),p.max(axis=0).astype(float)
            values=np.random.default_rng(918).uniform(lo-.1*(hi-lo),hi+.1*(hi-lo),(100000,3))
            points=wp.array(values,dtype=wp.vec3d)
            record=dict(case=case,queries=len(values),faces=len(f),query_sha256=hashlib.sha256(values.tobytes()).hexdigest(),seed=918)
        else:
            source_case='armadillo-graded' if case=='armadillo-small64' else case
            path=Path('results/validation/vtm/recovery-classification/recovery-classification-final')/(source_case+'-after')
            path=path/'mesh.npz' if path.is_dir() else path.with_suffix('.npz')
            stored=np.load(path)
            domain=warpower.SurfaceDomain(stored['input_points'],stored['input_faces'],surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
            owner=warpower.ExtractionWorkspace(domain,stored['points'],stored['full_tetrahedra'])
            e=owner
            args=[domain.tree,e.points,e.input_tetrahedra,e.input_count,e.tolerance,e.error,e.circumcenters,e.circumradii,e.signed_distance,e.selected,e._input_volume,e._input_quality]
            e.error.zero_()
            wp.launch(prepare_geometry,e.capacity,inputs=args,block_dim=32)
            assert e.error.numpy()[0]==0
            points=e.circumcenters
            if case=='armadillo-small64':
                points=wp.array(points.numpy()[:64],dtype=wp.vec3d)
            record=dict(case=case,input=str(path),input_sha256=sha(path),queries=len(points),faces=domain.tree.n,
                        query_sha256=hashlib.sha256(points.numpy().tobytes()).hexdigest())
        assert domain.error.numpy()[0]==0
        excluded=wp.full(len(points),-1,dtype=int)
        outputs={name:wp.zeros((len(points),3),dtype=int) for name in ['reference']+names}
        distances=wp.empty(len(points),dtype=wp.float64)
        graphs={}
        for name in ['reference']+names+['nearest']:
            with wp.ScopedCapture() as cap:
                if name=='nearest':
                    wp.launch(kernels.reference_nearest,len(points),inputs=[domain.tree,points,distances],block_dim=32)
                elif name=='reference':
                    wp.launch(kernels.reference_side,len(points),inputs=[domain.tree,points,excluded,outputs[name]],block_dim=32)
                else:
                    group=int(name.rsplit('_',1)[1])
                    wp.launch_tiled(getattr(kernels,name+'_side'),dim=(len(points)+(32//group)-1)//(32//group),inputs=[domain.tree,points,excluded,outputs[name]],block_dim=32)
            graphs[name]=cap.graph
            for _ in range(3):wp.capture_launch(cap.graph)
        wp.synchronize()
        initial=cpu_quota_snapshot()
        samples={name:[] for name in graphs}
        begin,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for repeat in range(15):
            order=list(graphs)
            if repeat%2:order.reverse()
            for name in order:
                wp.record_event(begin);wp.capture_launch(graphs[name]);wp.record_event(end)
                wp.synchronize_event(end)
                samples[name].append(wp.get_event_elapsed_time(begin,end))
        quota=cpu_quota_delta(initial,cpu_quota_snapshot())
        ref=outputs['reference'].numpy()[:,2]
        record.update(samples_ms=samples,median_ms={k:float(np.median(v)) for k,v in samples.items()},cpu_quota_delta=quota,
                      quota_clean=all(v['throttled_periods']==0 for v in quota.values()),
                      mismatches={k:int(np.count_nonzero(v.numpy()[:,2]!=ref)) for k,v in outputs.items()},
                      reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                      nearest_zero=int(np.count_nonzero(distances.numpy()==0)),
                      nearest_nonfinite=int(np.count_nonzero(~np.isfinite(distances.numpy()))))
        report['cases'].append(record)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(case,record['median_ms'],'mismatches',record['mismatches'],'clean',record['quota_clean'],flush=True)
    report['finished']=True
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
