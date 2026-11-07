"""Publish the validated negative #172 probes without runtime changes."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/prepared-hierarchy-probes'
baseline='ff5dff2ece6d2f55a0ee953da66b0d07190d528e'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==baseline
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for stem in ['prepared-hierarchy-box-existing','prepared-hierarchy-polynomial-existing','prepared-narrow-tree-1-existing','prepared-narrow-tree-8-existing']:
    assert '69 passed' in (w/(stem+'.log')).read_text()
for stem in ['prepared-hierarchy-mask-tests','prepared-narrow-tree-mask-tests']:
    assert '36 passed' in (w/(stem+'.log')).read_text()
for prefix in ['prepared-hierarchy','prepared-narrow-tree']:
    d=json.loads((w/(prefix+'-pilot-independent.json')).read_text())
    assert len(d)==4
    for row in d.values():
        assert row['after_ms']>row['before_ms']
        assert all(v for k,v in row['comparison'].items() if k!='maximum_coordinate_difference')
        assert all(row['work_equal'].values()) and row['inputs_parameters_equal']
        assert row['before_pool_bytes']==row['after_pool_bytes']
    commands=json.loads((w/(prefix+'-pilot/commands.json')).read_text())
    assert len(commands)==8 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in commands)
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
def copy(p,relative=None):
    q=out/(relative or p.name);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
    os.link(p,q);files.append(dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)))
sources=['prepared_hierarchy_queries.py','install_prepared_hierarchy.py','test-prepared-hierarchy-existing.py','test_prepared_hierarchy_masks.py','run-prepared-hierarchy.py','run-prepared-hierarchy-pilots.py','check-prepared-hierarchy-pilots.py','prepared_narrow_tree_queries.py','install_prepared_narrow_tree.py','test-prepared-narrow-tree-existing.py','test_prepared_narrow_tree_masks.py','run-prepared-narrow-tree.py','run-prepared-narrow-tree-pilots.py','check-prepared-narrow-tree-pilots.py','publish-prepared-hierarchy-probes.py']
for name in sources:copy(w/name)
logs=['prepared-hierarchy-box-existing','prepared-hierarchy-polynomial-existing','prepared-hierarchy-mask-tests','prepared-hierarchy-pilots','prepared-narrow-tree-1-existing','prepared-narrow-tree-8-existing','prepared-narrow-tree-mask-tests','prepared-narrow-tree-pilots']
for name in logs:copy(w/(name+'.log'))
for prefix in ['prepared-hierarchy','prepared-narrow-tree']:
    for extension in ['json','log']:copy(w/(prefix+'-pilot-independent.'+extension))
    for p in sorted((w/(prefix+'-pilot')).rglob('*')):
        if p.is_file():copy(p,p.relative_to(w))
helper=root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes()==(w/'check-large-reuse-proof-pairs.py').read_bytes()
manifest=dict(baseline=baseline,scope='Four private prepared-seed hierarchy layouts; all regress complete solves and remain unadopted. No runtime, public-test, standalone API or README change. No adoption memory/native audit or full extreme-box coverage claimed.',runtime_source_sha256={str(p):sha(p) for p in sorted(Path('warpower').glob('*.py'))},shared_helper=dict(path=str(helper.relative_to(root)),sha256=sha(helper)),reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance']},files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes')
