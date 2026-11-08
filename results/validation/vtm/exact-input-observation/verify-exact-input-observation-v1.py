"""Verify the observation publication and unchanged runtime before self-merge."""
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'results/validation/vtm/exact-input-observation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((out/'manifest.json').read_text());assert len(m['files'])==74
for r in m['files']:
 p,q=root/r['path'],root/r['original']
 assert sha(p)==sha(q)==r['sha256'] and p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0
for r in m['shared_references']:assert sha(root/r['path'])==r['sha256']
for name,digest in m['runtime_source_sha256'].items():
 assert sha(root/name)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',m['runtime']+':'+name])).hexdigest()==digest
assert not subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])
r=json.loads((out/'unchanged-input-study-v3-independent.json').read_text());assert r['complete']
assert len(r['cases'])==7
for name,c in r['cases'].items():
 assert all(c['work_equal'].values()) and all(c['geometry'][k] for k in ['full_tets_equal','material_tets_equal','roles_equal','oriented_boundary_equal'])
 if name!='thin-hand':assert c['geometry']['points_bitwise_equal']
 else:
  assert len(c['changed_coordinate_values'])==1 and c['geometry']['maximum_coordinate_difference']==7.995991022080595e-19
  assert c['independent']['full_hull']['valid'] and c['independent_control']['full_hull']['valid']
  assert c['independent']['boundary_self_intersection_pairs']==c['independent_control']['boundary_self_intersection_pairs']==0
a=json.loads((out/'unchanged-input-fixed16k-native-v1-attribution.json').read_text())
assert len(a['calls'])==227 and a['unchanged_calls']==3
assert abs(sum(x['span_ms'] for x in a['calls'] if x['unchanged'])-a['unchanged_reconstruction_span_ms'])<1e-9
assert 0<a['unchanged_span_fraction']<0.005
with np.load(out/'unchanged-input-fixed16k-native-v1.npz') as x,np.load(root/'.work/vtm/larger-prefix-final-v1/torus16384-fixed32-after.npz') as y:
 assert np.array_equal(x['points'].view(np.uint32),y['points'].view(np.uint32))
 assert np.array_equal(x['roles'],y['roles'])
def cells(x):
 y=np.sort(x,axis=1);return y[np.lexsort(y.T[::-1])]
with np.load(out/'unchanged-input-fixed16k-native-v1.npz') as x,np.load(root/'.work/vtm/larger-prefix-final-v1/torus16384-fixed32-after.npz') as y:
 for name in ['tetrahedra','full_tetrahedra']:assert np.array_equal(cells(x[name]),cells(y[name]))
 oriented=lambda t:sorted(min(tuple(v),tuple(np.roll(v,1)),tuple(np.roll(v,2))) for v in t)
 assert oriented(x['boundary'])==oriented(y['boundary'])
links=[]
for name in ['exact-input-observation.md','previous-complete-certificate.md']:
 p=root/'docs'/name
 for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('http://','https://','#')):continue
  q=(p.parent/target.split('#',1)[0]).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=name,target=rel))
result=dict(manifest_sha256=sha(out/'manifest.json'),artifact_hashes_aliases_readonly_verified=74,runtime_files_verified=len(m['runtime_source_sha256']),unchanged_runtime=True,standalone_paths_unchanged=True,readme_unchanged=True,observer_cases=7,native_current_output_verified=True,per_call_attribution_verified=True,links=links,verification_source_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444);print({k:v for k,v in result.items() if k!='links'})
