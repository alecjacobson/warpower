"""Publish the rejected #175 probe without runtime changes."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/prepared-split-probe'
baseline='b0fb1d46402922a546314e955bb5782f8b50a805'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==baseline
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for name,n in [('prepared-split-existing',69),('prepared-split-mask-tests',36),('prepared-split-dispatch-tests',6)]:
    assert f'{n} passed' in (w/(name+'.log')).read_text()
check=json.loads((w/'prepared-split-pilot-independent.json').read_text())
assert len(check)==2
for r in check.values():
    assert r['after_ms']>r['before_ms']
    assert all(v for k,v in r['comparison'].items() if k!='maximum_coordinate_difference')
    assert all(r['work_equal'].values()) and r['inputs_parameters_equal']
    assert r['independent']['full_hull']['valid'] and r['independent']['all_boundary_vertex_links_single_cycles']
commands=json.loads((w/'prepared-split-pilot/commands.json').read_text())
assert len(commands)==4 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in commands)
diag=json.loads((w/'prepared-split-diagnostic-4096.json').read_text())
assert sum(r['counts']['calls'] for r in diag['instances'])==624
assert sum(r['counts']['cells'] for r in diag['instances'])==12882912
assert all(r['counts']['fallback_lanes']==0 and r['counts']['fallback_cells']==0 for r in diag['instances'])
eq=json.loads((w/'prepared-split-diagnostic-equivalence.json').read_text())
assert all(v for k,v in eq.items() if k!='maximum_coordinate_difference')
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
def copy(p,relative=None):
    q=out/(relative or p.name);q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q)
    files.append(dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)))
sources=['prepared_split_queries.py','test-prepared-split-existing.py','test_prepared_split_masks.py','test_prepared_split_dispatch.py','run-prepared-split.py','run-prepared-split-pilots.py','check-prepared-split-pilots.py','prepared-split-properties.py','profile_prepared_split.py','publish-prepared-split-probe.py']
for name in sources:copy(w/name)
for name in ['prepared-split-existing.log','prepared-split-mask-tests.log','prepared-split-dispatch-tests.log','prepared-split-properties.log','prepared-split-properties.json','prepared-split-pilots.log','prepared-split-pilot-independent.log','prepared-split-pilot-independent.json','prepared-split-diagnostic-4096.log','prepared-split-diagnostic-4096.json','prepared-split-diagnostic-4096.npz','prepared-split-diagnostic-run-4096.json','prepared-split-diagnostic-equivalence.json']:copy(w/name)
for p in sorted((w/'prepared-split-pilot').iterdir()):
    if p.is_file():copy(p,p.relative_to(w))
helper=root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes()==(w/'check-large-reuse-proof-pairs.py').read_bytes()
manifest=dict(baseline=baseline,scope='Private prepared-seed strict/exact lane split is slower at both tested sizes and remains unadopted. No runtime, public-test, standalone API or README change. No broad adoption memory/native audit claimed.',runtime_source_sha256={str(p):sha(p) for p in sorted(Path('warpower').glob('*.py'))},shared_helper=dict(path=str(helper.relative_to(root)),sha256=sha(helper)),reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance']},files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes')
