import json
import sys
from pathlib import Path

ROOT = Path('/home/horde/projects/power-diagram')
sys.path[:0] = [str(ROOT/'experiments/vorotracing-integration'), str(ROOT/'experiments/radfoam-integration')]
import torch
import warp_renderer as wr
from benchmark_renderer import timed
from upstream import load
from spatial import morton_permutation, permute_graph
import numerics

upstream = load('native')
from vorotracing import cuda
from vorotracing.datasets.datasets import ColmapConfig
numerics.instrument_native(cuda._wrapper)
torch.manual_seed(2026)
out = ROOT/'results/local/vorotracing-20261110/training-block-tuning.json'
model = upstream.VoroTracing.from_pretrained(ROOT/'results/local/vorotracing-20261110/native-full/model/model.pt', upstream.VoroTracingConfig())
with torch.no_grad():
    perm = morton_permutation(model.primal_points)
    model.point_adjacency, model.point_adjacency_offsets = permute_graph(model.point_adjacency, model.point_adjacency_offsets, perm)
    for name in ['primal_points', 'density', 'att_diffuse', 'att_specular']:
        setattr(model, name, torch.nn.Parameter(getattr(model,name)[perm].contiguous()))
    model.bvh.build(model.primal_points)
dataset = ColmapConfig(data_path=str(ROOT/'data/radfoam-training'), scene='bonsai', init_points='original').load('train', 2)
rays = dataset.sample_batch(1000000)['rays']
starts = model.get_starting_point(rays)
density = model.density.exp().detach().requires_grad_()
params = [model.primal_points, model.att_diffuse, model.att_specular, density]
original_launch = wr.wp.launch
selection = [256,256]

def launch(kernel, *args, **kwargs):
    if kernel is wr.trace_forward: kwargs['block_dim'] = selection[0]
    if kernel is wr.trace_backward: kwargs['block_dim'] = selection[1]
    return original_launch(kernel, *args, **kwargs)

wr.wp.launch = launch

def replay(backend):
    for p in params: p.grad = None
    common = [*params, model.point_adjacency, model.point_adjacency_offsets, rays, starts]
    if backend == 'native':
        rgba, _, _, _, _, dist = cuda.TraceVoroTracing.apply(*common, None, .001, 1024, False)
    else:
        rgba, dist, _ = wr.Trace.apply(*common, .001, 1024)
    (rgba.mean()+.002*dist.mean()).backward()

configs = [(256,256),(128,128),(64,64),(256,128),(128,64),(64,128),(128,256)]
report = {'complete':False, 'scope':'Same full 2m-site native checkpoint with identical Morton-relabeled graph and one million uniformly sampled rays from all 255 training cameras. Timed FP32 forward + backward + gradient initialization and common counted sanitization. No optimizer or ray generation.', 'samples':[], 'native':[]}
for round_index in range(2):
    native = timed(lambda: replay('native'), 8)
    report['native'].append(native)
    print('native',round_index,native,flush=True)
    for config in configs if round_index==0 else configs[::-1]:
        selection[:] = config
        row = {'round':round_index,'forward_block':config[0],'backward_block':config[1],**timed(lambda: replay('warp'),8)}
        report['samples'].append(row)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
report.update(complete=True,numerics=numerics.snapshot())
out.write_text(json.dumps(report,indent=2)+'\n')
