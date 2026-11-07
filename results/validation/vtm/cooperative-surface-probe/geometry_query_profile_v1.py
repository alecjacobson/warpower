"""Static current-mesh kernel profile for #159; not complete-solve evidence."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from warpower import _extraction
from warpower._surface_bvh import SurfaceTree
from warpower._wide_center import uncertain
from extraction_split import prepare_geometry, query_surface

wp.set_module_options({'enable_backward': False, 'fast_math': False})


@wp.kernel
def query_selection(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
    distances: wp.array[wp.float64], keep: wp.array[int],
    volumes: wp.array[wp.float64], quality: wp.array[wp.float64],
):
    i = wp.tid()
    if i >= count[0] or error[0] != 0:
        return
    status = _extraction.selection_center(tree, centers[i], radii[i], tolerance)
    if status.error != 0:
        wp.atomic_or(error, 0, status.error)
        return
    distances[i] = status.distance_sign
    keep[i] = status.retained


@wp.kernel
def geometry_categories(points: wp.array[wp.vec3], tets: wp.array2d[int], result: wp.array[int]):
    i = wp.tid()
    p = wp.vec3d(points[tets[i, 0]])
    a = wp.vec3d(points[tets[i, 1]]) - p
    b = wp.vec3d(points[tets[i, 2]]) - p
    c = wp.vec3d(points[tets[i, 3]]) - p
    determinant = wp.dot(a, wp.cross(b, c))
    wp.atomic_add(result, int(uncertain(a, b, c, determinant)), 1)


root = Path.cwd()
w = root / '.work/vtm'
out = w / 'geometry-query-profile-v1'
assert not out.exists()
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert revision == '305758fac0575cfb4c7ded604df02a52560905ce'
out.mkdir()
wp.load_module(module='extraction_split', device='cuda:0', block_dim=32)
wp.load_module(module=__name__, device='cuda:0', block_dim=32)
report = dict(
    scope='Static final-mesh classification kernels only. All inputs are fixed current validated meshes. Includes error reset in each graph; geometry preparation is included only in full/prepare/split paths. Query-only rows intentionally reuse immutable prepared geometry. Not whole-mesher timings, no setup comparison, and no adoption claim. Invalid-input/failure semantics of any future split remain unvalidated.',
    runtime=revision, cases={},
    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), w/'extraction_split.py', root/'warpower/_extraction.py']},
)


def run_case(name):
    base = root / 'results/validation/vtm/recovery-classification/recovery-classification-final' / (name+'-after')
    path = base/'mesh.npz' if base.is_dir() else base.with_suffix('.npz')
    stored = np.load(path)
    domain = warpower.SurfaceDomain(stored['input_points'], stored['input_faces'], surface_subdivisions=1, crease_subdivisions=1, feature_angle=120)
    assert domain.error.numpy()[0] == 0
    e = warpower.ExtractionWorkspace(domain, stored['points'], stored['full_tetrahedra'])
    args = [domain.tree,e.points,e.input_tetrahedra,e.input_count,e.tolerance,e.error,e.circumcenters,e.circumradii,e.signed_distance,e.selected,e._input_volume,e._input_quality]
    categories = wp.zeros(2,dtype=int,device=e.device)
    wp.launch(geometry_categories,e.capacity,inputs=[e.points,e.input_tetrahedra,categories],device=e.device,block_dim=32)
    variants = [
        ('full',[_extraction.classify]),
        ('prepare_only',[prepare_geometry]),
        ('full_query_only',[query_surface]),
        ('full_split',[prepare_geometry,query_surface]),
        ('private',[_extraction.classify_selection]),
        ('private_query_only',[query_selection]),
        ('private_split',[prepare_geometry,query_selection]),
    ]
    graphs = {}
    for label,kernels in variants:
        with wp.ScopedCapture() as cap:
            e.error.zero_()
            for kernel in kernels:
                wp.launch(kernel,e.capacity,inputs=args,device=e.device,block_dim=32)
        graphs[label] = cap.graph
    wp.synchronize()
    result=dict(input=str(path.relative_to(root)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sites=len(e.points),tetrahedra=e.capacity,geometry_categories=dict(zip(['ordinary','uncertain_exact_fallback'],categories.numpy().tolist())),rows={})
    full_reference=None;private_reference=None
    begin,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
    for label,_ in variants:
        # Preparation is deliberately outside query-only timing and refreshed
        # immediately before each such variant. Sites/domain do not change.
        if label.endswith('query_only'):
            wp.capture_launch(graphs['prepare_only'])
        for _ in range(3):wp.capture_launch(graphs[label])
        wp.synchronize()
        initial=cpu_quota_snapshot();samples=[]
        for _ in range(20):
            wp.record_event(begin);wp.capture_launch(graphs[label]);wp.record_event(end)
            wp.synchronize_event(end);samples.append(wp.get_event_elapsed_time(begin,end))
        quota=cpu_quota_delta(initial,cpu_quota_snapshot())
        row=dict(gpu_ms=samples,median_ms=float(np.median(samples)),cpu_quota_delta=quota,quota_clean=all(v['throttled_periods']==0 for v in quota.values()),error=int(e.error.numpy()[0]))
        assert row['error']==0 and row['quota_clean'],(name,label,row)
        values={key:getattr(e,key).numpy() for key in ['circumcenters','circumradii','signed_distance','selected','_input_volume','_input_quality']}
        if label=='full':full_reference=values
        elif label=='private':
            private_reference=values
            for key in values:
                if key!='signed_distance':np.testing.assert_array_equal(values[key],full_reference[key])
            np.testing.assert_array_equal(values['signed_distance']<=0,full_reference['signed_distance']<=0)
        elif label!='prepare_only':
            reference=private_reference if label.startswith('private') else full_reference
            for key in values:np.testing.assert_array_equal(values[key],reference[key])
            row['all_fields_bitwise_equal_to_fused']=True
        row['output_sha256']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in values.items()}
        result['rows'][label]=row
        report['cases'][name]=result
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,label,row['median_ms'],flush=True)
    return result


for name in ['torus1000','torus4096','armadillo-uniform','armadillo-graded']:
    report['cases'][name]=run_case(name)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
