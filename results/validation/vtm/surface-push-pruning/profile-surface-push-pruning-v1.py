"""Static complete classifier/cache and padded quadrature costs; no solve claim."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from surface_push_pruning_install_v1 import install
import surface_push_pruning_v1 as nearest
from warpower import SurfaceDomain, _extraction, _surface_query_cache as query
from warpower._ordered_surface_queries import OrderedSurfaceQueryCache, ORDERED
from warpower._surface_bvh import Closest

parser=argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--phase', choices=['before','after'], required=True)
args=parser.parse_args(); assert not args.output.exists()
if args.phase=='after': install()
with np.load(args.input) as d:
    ip=d['input_points']; faces=d['input_faces']; sites=d['points']; cells=d['full_tetrahedra']; boundary=d['boundary']
domain=SurfaceDomain(ip,faces,surface_subdivisions=1,crease_subdivisions=1,feature_angle=120)
assert domain.error.numpy()[0]==0
capacity=8*len(sites); assert len(cells)<=capacity
padded=np.full((capacity,4),-7,np.int32); padded[:len(cells)]=cells
points=wp.array(sites,dtype=wp.vec3); tets=wp.array(padded,dtype=int); count=wp.full(1,len(cells),dtype=int)
paths=[Path(__file__),Path(nearest.__file__),Path(__file__).with_name('surface_push_pruning_install_v1.py'),*Path('warpower').glob('*.py')]
report=dict(scope=__doc__,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),phase=args.phase,input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},sites=len(sites),cells=len(cells),capacity=capacity,face_capacity=4*capacity,rows=[],complete=False,limits='Static validated final meshes. Setup and host reads excluded. Classifier rows include ordering, sorting, cache rebuild, and cold reset when selected. Full reserved capacities are launched. Nearest-only rows use live cell centers.')
start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)

def measure(name,graph,arrays,properties):
    for _ in range(5): wp.capture_launch(graph)
    wp.synchronize_device(domain.device)
    before=cpu_quota_snapshot(); samples=[]
    for _ in range(20):
        wp.record_event(start); wp.capture_launch(graph); wp.record_event(end); wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(start,end))
    quota=cpu_quota_delta(before,cpu_quota_snapshot())
    values={name:value.numpy() for name,value in arrays.items()}
    assert 'error' not in values or values['error'][0]==0
    hashes={}
    for key,value in values.items():
        if value.dtype.names:
            for field in value.dtype.names: hashes[key+'.'+field]=hashlib.sha256(value[field].tobytes()).hexdigest()
        else: hashes[key]=hashlib.sha256(value.tobytes()).hexdigest()
    row=dict(name=name,gpu_ms=samples,median_ms=float(np.median(samples)),cpu_quota_delta=quota,quota_clean=all(v['throttled_periods']==0 for v in quota.values()),output_sha256=hashes,properties=properties)
    if name=='quadrature': row['values']={k:float(v.item()) for k,v in values.items()}
    report['rows'].append(row); args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['median_ms'],row['quota_clean'],flush=True)
    assert row['quota_clean']

centers=None
for mode,kernel in [('full',query.classify_cached),('selection',query.selection_cached),('retained',query.retained_cached)]:
    cache=OrderedSurfaceQueryCache(domain,capacity)
    arrays=dict(error=wp.zeros(1,dtype=int),keep=wp.full(capacity,77,dtype=int),quality=wp.full(capacity,77,dtype=wp.float64))
    inputs=[domain.tree,points,tets,count,0.4,arrays['error']]
    if mode=='retained': inputs += [arrays['keep'],arrays['quality']]
    else:
        arrays.update(center=wp.full(capacity,wp.vec3d(77),dtype=wp.vec3d),radius=wp.full(capacity,77,dtype=wp.float64),distance=wp.full(capacity,77,dtype=wp.float64),volume=wp.full(capacity,77,dtype=wp.float64))
        inputs += [arrays[k] for k in ['center','radius','distance','keep','volume','quality']]
    observed=dict(arrays)
    for field in query.FIELDS: observed['probe.'+field]=getattr(cache.current,field)
    properties={'unordered':wp.get_cuda_kernel_properties(kernel,device=domain.device,block_dim=32)}
    if cache._has_order_storage: properties['ordered']=wp.get_cuda_kernel_properties(ORDERED[kernel],device=domain.device,block_dim=32)
    for temperature in ['cold','warm']:
        with wp.ScopedCapture(device=domain.device) as capture:
            arrays['error'].zero_()
            if temperature=='cold': cache.reset()
            cache.launch(kernel,capacity,inputs)
        measure(mode+'-'+temperature,capture.graph,observed,properties)
    if mode=='full': centers=arrays['center']

limits=wp.full(len(cells),1e150,dtype=wp.float64); hits=wp.empty(len(cells),dtype=Closest)
kernel=nearest.reference if args.phase=='before' else nearest.candidate
wp.load_module(module=nearest,device=domain.device,block_dim=32)
with wp.ScopedCapture(device=domain.device) as capture:
    wp.launch(kernel,len(cells),inputs=[domain.tree,centers,limits,hits],block_dim=32)
measure('nearest-live',capture.graph,{'hits':hits},wp.get_cuda_kernel_properties(kernel,device=domain.device,block_dim=32))

face_capacity=4*capacity
padded=np.full((face_capacity,3),-7,np.int32); padded[:len(boundary)]=boundary
bf=wp.array(padded,dtype=int); bc=wp.full(1,len(boundary),dtype=int)
arrays=dict(integral=wp.zeros(1,dtype=wp.float64),maximum=wp.zeros(1,dtype=wp.float64),error=wp.zeros(1,dtype=int))
wp.load_module(module=_extraction,device=domain.device,block_dim=32)
with wp.ScopedCapture(device=domain.device) as capture:
    for value in arrays.values(): value.zero_()
    wp.launch(_extraction.surface_error,3*face_capacity,inputs=[domain.tree,points,bf,bc,*arrays.values()],block_dim=32)
measure('quadrature',capture.graph,arrays,wp.get_cuda_kernel_properties(_extraction.surface_error,device=domain.device,block_dim=32))
report['complete']=True; args.output.write_text(json.dumps(report,indent=2)+'\n')
