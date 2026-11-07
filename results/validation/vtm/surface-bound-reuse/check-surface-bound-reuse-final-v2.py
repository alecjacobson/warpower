"""Independent final meshes and standalone nonregression evidence."""
import gzip,hashlib,json,subprocess,shutil,tempfile
from pathlib import Path
import numpy as np
root=Path.cwd();w=root/'.work/vtm'
helper=(w/'check-large-reuse-proof-pairs.py').read_text().split("out=w/'large-reuse-proof-pairs';summary={}")[0]
scratch=Path('/opt/warpower-generated-cache/validation-derived');scratch.mkdir(parents=True,exist_ok=True)
helper=helper.replace("dir=w)","dir=scratch)")
helper=helper.replace("NativeSurfaceDistance(w/'surface-distance',w)","NativeSurfaceDistance(w/'surface-distance',scratch)")
helper=helper.replace("xml=ET.parse(vtu)","xml=ET.parse(gzip.open(vtu,'rb') if str(vtu).endswith('.gz') else vtu)")
helper=helper.replace("str(path.parent/'boundary.off')","str(boundary_path)")
helper=helper.replace("  text=subprocess.run(","  boundary_path=path.parent/'boundary.off'\n  if not boundary_path.exists():\n   boundary_path=Path(tmp)/'boundary.off'\n   boundary_path.write_bytes(gzip.decompress((path.parent/'boundary.off.gz').read_bytes()))\n  text=subprocess.run(")
# validate()'s earlier TemporaryDirectory closes before the VTU section.
helper=helper.replace("boundary_path=Path(tmp)/'boundary.off'", "boundary_path=scratch/'current-boundary.off'")
exec(compile(helper,str(w/'check-large-reuse-proof-pairs.py'),'exec'))
out=w/'surface-bound-reuse-final';commands=json.loads((out/'commands.json').read_text())
assert len(commands)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
def paths(stem):
    return (stem/'result.json',stem/'mesh.npz',stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'),stem.with_suffix('.npz'),None)
def timing(d):return d.get('median_gpu_milliseconds',d.get('median_additional_replay_ms'))
def parameters(d):return {k:v for k,v in d['parameters'].items() if k not in ['save_mesh','dot','output']}
summary=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline='18ecb03c10cb10e42a5089e8253ed92a24c3bae1',cases={},standalone={})
pairs=[(name,out/(name+'-before')) for name in ['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','armadillo-uniform','armadillo-graded']]
for name,before in pairs:
    ap,an,av=paths(before);bp,bn,bv=paths(out/(name+'-after'))
    a,b=[json.loads(p.read_text()) for p in [ap,bp]]
    for d in [a,b]:quota(d);assert d['valid']==1 and d['error']==0
    assert parameters(a)==parameters(b),(name,parameters(a),parameters(b))
    x,y=np.load(an),np.load(bn);comparison=compare(x,y)
    assert all(comparison[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,comparison)
    if name not in ['thin-hand','cavity','disconnected']:assert comparison['points_bitwise_equal'],(name,comparison)
    assert comparison['maximum_coordinate_difference'] <= 1e-8,(name,comparison)
    inputs={k:bool(np.array_equal(x[k],y[k])) for k in ['input_points','input_faces']};assert all(inputs.values())
    changed=np.argwhere(x['points'].view(np.uint32)!=y['points'].view(np.uint32))
    values=[[int(i),int(j),float(x['points'][i,j]),float(y['points'][i,j])] for i,j in changed]
    keys=['counts','rounds','refinement_rejections','quality_rejections']+(['optimizer_steps','quality_moves'] if 'optimizer_steps' in a else ['steps','moves'])
    if 'reconstruction_routing' in a or 'reconstruction_routing' in b:keys.append('reconstruction_routing')
    work={k:a[k]==b[k] for k in keys};assert all(work.values()),(name,work)
    row=dict(before_result=str(ap.relative_to(root)),after_result=str(bp.relative_to(root)),before_sha256=hashlib.sha256(an.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(bn.read_bytes()).hexdigest(),before_ms=timing(a),after_ms=timing(b),speedup=timing(a)/timing(b),comparison=comparison,changed_coordinate_values=values,input_arrays_equal=inputs,work_equal=work,parameters_equal=True,all_first_and_timed_quotas_clean=True,independent=validate(bn,bv))
    for phase,d in [('before',a),('after',b)]:
        row[phase+'_setup_seconds']=d['setup_seconds'];row[phase+'_first_replay_ms']=d.get('first_replay_gpu_milliseconds',d.get('first_replay_gpu_ms'))
        if 'persistent_workspace_pool_bytes' in d:row[phase+'_pool_bytes']=d['persistent_workspace_pool_bytes']
    # The local bound stack adds no persistent device arrays.
    summary['cases'][name]=row
    (w/'surface-bound-reuse-final-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,row['before_ms'],row['after_ms'],comparison,flush=True)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
    a,b=[json.loads((out/(name+'-'+phase+'.json')).read_text()) for phase in ['before','after']]
    if 'full' in a:
        for d in [a,b]:
            assert all(v['throttled_periods']==0 for v in d['full']['cpu'].values()) and d['error']==0
        skip={'first_replay_ms','full','base','auxiliary','certificate'}
        unchanged={k:a[k]==b[k] for k in a if k not in skip};assert all(unchanged.values()),(name,unchanged)
        am,bm=a['full']['median_ms'],b['full']['median_ms']
    else:
        keys=['points_sha256','weights_sha256','distribution','n','seed','weighted','nnz','status_counts','diagnostics']
        unchanged={k:a[k]==b[k] for k in keys};assert all(unchanged.values()),(name,unchanged)
        assert not a['cpu_throttled'] and not b['cpu_throttled']
        am,bm=1000*a['median_seconds'],1000*b['median_seconds']
    summary['standalone'][name]=dict(before_ms=am,after_ms=bm,relative_change=bm/am-1,unchanged=unchanged,all_timed_quotas_clean=True)
(w/'surface-bound-reuse-final-independent.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['standalone'],indent=2),flush=True)
