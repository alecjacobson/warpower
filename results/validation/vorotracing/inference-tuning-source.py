"""Compare indexing and launch variants without changing the fitted renderer."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/home/horde/projects/power-diagram')
DRIVER = ROOT/'experiments/vorotracing-integration'
OUT = ROOT/'results/local/vorotracing-20261110'
sys.path.insert(0, str(DRIVER))
import torch
import warp as wp
from benchmark_renderer import timed
from upstream import load
from validate_renderer import difference

upstream = load('native')
from vorotracing.datasets.datasets import ColmapConfig

variants_dir = ROOT/'.work/vorotracing-20261110/inference-variants'
variants_dir.mkdir(exist_ok=True)
source = (DRIVER/'warp_renderer.py').read_text()
modules = {}
for layout in ['matrix', 'flat']:
    for registers in [None, 48, 64, 80]:
        name = f'{layout}_reg{registers}'
        code = source
        if layout == 'flat':
            code = code.replace('wp.array2d[wp.uint64]', 'wp.array[wp.uint64]')
            for i in range(4):
                code = code.replace(f'wp.cast(t[cell, s.ids[{i}]], vec4h)', f'wp.cast(t[cell * 64 + s.ids[{i}]], vec4h)')
            code = code.replace('.view(torch.int64).reshape(-1, 64)', '.view(torch.int64).reshape(-1)')
        if registers is not None:
            code = code.replace('@wp.kernel\ndef trace_inference(', f'@wp.kernel(cuda_max_registers={registers})\ndef trace_inference(')
        path = variants_dir/(name+'.py')
        path.write_text(code)
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module

model = upstream.VoroTracing.from_pretrained(OUT/'native-full/model/model.pt', upstream.VoroTracingConfig())
renderer = upstream.VoroTracingInfer(model)
dataset = ColmapConfig(data_path=str(ROOT/'data/radfoam-training'),scene='bonsai',init_points='original').load('test',2)
report = {'complete':False,'protocol':'Same full 2m-site native checkpoint, FP16 paper settings, identical native starting cells and Morton graph; fixed inputs, traversal only; native render with starting cells also includes cached tile dispatch. Variants run in forward and reverse order.', 'views':[]}
launch_original = wp.launch
block = 128

def launch(kernel,*args,**kwargs):
    if kernel.key == 'trace_inference':
        kwargs['block_dim'] = block
    return launch_original(kernel,*args,**kwargs)

wp.launch = launch

def graphed_timing(call):
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured_output = call()
    timing = timed(graph.replay,20)
    del captured_output, graph
    return timing

def prepared_call(module, rays, start, permutation, flat):
    # Retain only model-constant Warp views; frame inputs still bind on each call.
    static = [
        module.tensor(renderer.primal_points, wp.vec3),
        module.tensor(renderer.diffuse_fp16.view(torch.int64).reshape(-1) if flat else renderer.diffuse_fp16.view(torch.int64).reshape(-1,64), wp.uint64),
        module.tensor(renderer.specular_fp16.view(torch.int64).reshape(-1) if flat else renderer.specular_fp16.view(torch.int64).reshape(-1,64), wp.uint64),
        module.tensor(renderer.density.reshape(-1)),
        module.tensor(renderer.point_adjacency),
        module.tensor(renderer.point_adjacency_offsets),
        module.tensor(renderer.adjacent_diff.view(torch.int64).reshape(-1),wp.uint64),
    ]
    def call():
        rgba = torch.empty((*rays.shape[:-1],4),device=rays.device,dtype=torch.float16)
        wp.launch(module.trace_inference,rays.numel()//6,inputs=static+[
            module.tensor(rays.reshape(-1,2,3),wp.vec3),module.tensor(start.reshape(-1)),module.tensor(permutation),.01,1024,.001,module.tensor(rgba.reshape(-1,4),module.vec4h)
        ],stream=module.stream(),block_dim=128)
        return rgba
    return call

with torch.no_grad():
    for view in [0,10,36]:
        rays = dataset[view]['rays'].cuda()
        start = renderer.get_starting_point(rays)
        permutation = renderer._tile_perm(*rays.shape[:-1])
        common = [renderer.primal_points, renderer.diffuse_fp16, renderer.specular_fp16, renderer.density, renderer.point_adjacency, renderer.point_adjacency_offsets, rays, start, renderer.adjacent_diff, permutation, .01, 1024, .001]
        def native():
            return renderer.render(rays,start_point=start,weight_threshold=.01,cell_skip_threshold=.001)
        ref = native()
        row = {'view':view,'native':[],'native_graph':graphed_timing(native),'samples':[]}
        configs = [(name,size) for name in modules for size in ([64,128,256] if name.endswith('None') else [128])]
        for round_index in range(2):
            row['native'].append(timed(native,20))
            for name,size in configs if round_index==0 else configs[::-1]:
                block = size
                module = modules[name]
                call = lambda: module.infer(*common)
                got = call()
                parity = difference(got,ref)
                sample = {'variant':name,'block':size,'round':round_index,'parity':parity,**timed(call,20)}
                if round_index == 0:
                    sample['graph'] = graphed_timing(call)
                row['samples'].append(sample)
                print(view,json.dumps(sample),flush=True)
        report['views'].append(row)
        for name in ['matrix_regNone','flat_regNone']:
            block = 128
            call = prepared_call(modules[name],rays,start,permutation,name.startswith('flat'))
            row['samples'].append({'variant':name+'_prepared','block':128,'round':0,'parity':difference(call(),ref),**timed(call,20),'graph':graphed_timing(call)})
        (OUT/'inference-tuning.json').write_text(json.dumps(report,indent=2)+'\n')
report['complete']=True
(OUT/'inference-tuning.json').write_text(json.dumps(report,indent=2)+'\n')
