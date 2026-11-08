"""Static surface-error cost with production-sized reserved faces; no full-solve claim."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from warpower import SurfaceDomain, _extraction
import surface_sample_parallel_v1 as candidate

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();assert not args.output.exists()
with np.load(args.input) as stored:
    input_points=stored['input_points'];input_faces=stored['input_faces'];sites=stored['points'];boundary=stored['boundary']
domain=SurfaceDomain(input_points,input_faces,surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
assert domain.error.numpy()[0]==0
# The complete mesher reserves 8*N tetrahedra, hence 32*N boundary faces.
capacity=32*len(sites)
assert capacity>=len(boundary)
padded=np.full((capacity,3),-7,np.int32);padded[:len(boundary)]=boundary
points=wp.array(sites,dtype=wp.vec3);faces=wp.array(padded,dtype=int);count=wp.full(1,len(boundary),dtype=int)
variants={'original32':(_extraction.surface_error,1,32),'samples32':(candidate.surface_error_samples,3,32),'samples64':(candidate.surface_error_samples,3,64),'samples128':(candidate.surface_error_samples,3,128)}
graphs={};outputs={};properties={}
for name,(kernel,factor,block) in variants.items():
    wp.load_module(module=kernel.module,device=domain.device,block_dim=block)
    properties[name]=wp.get_cuda_kernel_properties(kernel,device=domain.device,block_dim=block)
    integral=wp.zeros(1,dtype=wp.float64);maximum=wp.zeros(1,dtype=wp.float64);error=wp.zeros(1,dtype=int)
    outputs[name]=[integral,maximum,error]
    with wp.ScopedCapture(device=domain.device) as capture:
        integral.zero_();maximum.zero_();error.zero_()
        wp.launch(kernel,factor*capacity,inputs=[domain.tree,points,faces,count,integral,maximum,error],block_dim=block)
    graphs[name]=capture.graph
paths=[Path(__file__),Path(candidate.__file__),Path('warpower/_extraction.py'),Path('warpower/_surface_bvh.py')]
report=dict(scope=__doc__,baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},sites=len(sites),surface_triangles=len(input_faces),boundary_faces=len(boundary),face_capacity=capacity,properties=properties,rows=[],complete=False,limits='Static successful final meshes; no complete trajectory or adoption. Reset/launch/capacity costs included; setup and host checks excluded. Atomic summation order may differ.')
start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
reference=None
for name,graph in graphs.items():
    for _ in range(5):wp.capture_launch(graph)
    wp.synchronize_device(domain.device)
    before=cpu_quota_snapshot();samples=[]
    for _ in range(20):
        wp.record_event(start);wp.capture_launch(graph);wp.record_event(end);wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(start,end))
    quota=cpu_quota_delta(before,cpu_quota_snapshot())
    value=[a.numpy().item() for a in outputs[name]]
    assert value[2]==0
    if reference is None:reference=value
    else:np.testing.assert_allclose(value[:2],reference[:2],rtol=2e-12,atol=0)
    row=dict(variant=name,gpu_ms=samples,median_ms=float(np.median(samples)),cpu_quota_delta=quota,quota_clean=all(v['throttled_periods']==0 for v in quota.values()),integral=value[0],maximum=value[1],error=value[2],relative_integral_difference=(value[0]-reference[0])/reference[0] if reference[0] else 0,maximum_bitwise_equal=np.float64(value[1]).tobytes()==np.float64(reference[1]).tobytes())
    report['rows'].append(row);args.output.write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True);assert row['quota_clean']
report['complete']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
