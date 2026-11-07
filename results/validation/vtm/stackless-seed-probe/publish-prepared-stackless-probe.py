"""Publish the neutral #177 probe without runtime changes."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/stackless-seed-probe'
baseline='ea1c00a0fdd6d920ae755b59028cbeb5c311cea7'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==baseline
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for name,n in [('prepared-stackless-existing',69),('prepared-stackless-new-tests',76),('prepared-stackless-flat-masks',36)]:
    assert f'{n} passed' in (w/(name+'.log')).read_text()
check=json.loads((w/'prepared-stackless-pilot-independent.json').read_text())
assert len(check)==2
for r in check.values():
    assert abs(r['after_ms']/r['before_ms']-1)<0.003
    assert all(v for k,v in r['comparison'].items() if k!='maximum_coordinate_difference')
    assert all(r['work_equal'].values()) and r['inputs_parameters_equal']
    assert r['independent']['full_hull']['valid'] and r['independent']['all_boundary_vertex_links_single_cycles']
commands=json.loads((w/'prepared-stackless-pilot/commands.json').read_text())
assert len(commands)==4 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in commands)
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
def copy(p,relative=None):
    q=out/(relative or p.name);q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q)
    files.append(dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)))
sources=['prepared_stackless_queries.py','test-prepared-stackless-existing.py','test_prepared_stackless_masks.py','test_prepared_stackless_flat_masks.py','test_prepared_stackless_traversal.py','run-prepared-stackless.py','run-prepared-stackless-pilots.py','check-prepared-stackless-pilots.py','publish-prepared-stackless-probe.py']
for name in sources:copy(w/name)
for name in ['prepared-stackless-existing.log','prepared-stackless-new-tests.log','prepared-stackless-flat-masks.log','prepared-stackless-pilots.log','prepared-stackless-pilot-independent.log','prepared-stackless-pilot-independent.json']:copy(w/name)
for p in sorted((w/'prepared-stackless-pilot').iterdir()):
    if p.is_file():copy(p,p.relative_to(w))
helper=root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes()==(w/'check-large-reuse-proof-pairs.py').read_bytes()
manifest=dict(baseline=baseline,scope='Private existing-segment stackless traversal is effectively neutral at 1k/4k and remains unadopted. No runtime, public-test, standalone API or README change. No broad adoption memory/native audit claimed.',runtime_source_sha256={str(p):sha(p) for p in sorted(Path('warpower').glob('*.py'))},shared_helper=dict(path=str(helper.relative_to(root)),sha256=sha(helper)),reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance']},files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes')
