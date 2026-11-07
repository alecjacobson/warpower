"""Publish the unadopted #182 experiments, retaining all observed variants."""
import hashlib, json, os, subprocess
from pathlib import Path
import numpy as np

root = Path.cwd(); w = root / '.work/vtm'
out = root / 'results/validation/vtm/previous-complete-certificate'
baseline = 'cc0ad8fb935871031c2b082c5eaedbcf2d866992'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == baseline
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for name, count in [('previous-complete-existing',69), ('previous-complete-replay-tests',15),
                    ('previous-staged-existing',69), ('previous-staged-replay-tests',23)]:
    assert f'{count} passed' in (w / (name + '.log')).read_text()
for variant in ['complete','staged']:
    checked = json.loads((w/f'previous-{variant}-pilot-independent.json').read_text())
    assert len(checked) == 2
    for row in checked.values():
        assert all(v for k,v in row['comparison'].items() if k != 'maximum_coordinate_difference')
        assert all(row['work_equal'].values()) and row['inputs_parameters_equal']
        assert row['independent']['full_hull']['valid']
        assert row['independent']['all_boundary_vertex_links_single_cycles']
    commands = json.loads((w/f'previous-{variant}-pilot/commands.json').read_text())
    assert len(commands) == 4
    assert all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in commands)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(t):
    t=np.sort(t,axis=1); return t[np.lexsort(t.T[::-1])]

shared = []
for n in [1000,4096]:
    ref = root / f'results/validation/vtm/bounded-predicate-filters/float32-final/torus{n}-after.npz'
    x,y = np.load(ref), np.load(w/f'previous-complete-observer-{n}.npz')
    for key in ['points','roles','input_points','input_faces']:
        assert np.array_equal(x[key],y[key]),key
    for key in ['full_tetrahedra','tetrahedra']:
        assert np.array_equal(canonical(x[key]),canonical(y[key]))
    shared.append(dict(path=str(ref.relative_to(root)),sha256=sha(ref)))
    ref = ref.with_suffix('.json')
    x,y = [json.loads(p.read_text()) for p in [ref,w/f'previous-complete-observer-run-{n}.json']]
    for key in ['counts','rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections']:
        assert x[key] == y[key],key
    shared.append(dict(path=str(ref.relative_to(root)),sha256=sha(ref)))

helper = root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes() == (w/helper.name).read_bytes()
shared.append(dict(path=str(helper.relative_to(root)),sha256=sha(helper)))
assert not out.exists(); out.mkdir(parents=True)
files=[]
def copy(p):
    q=out/p.relative_to(w); q.parent.mkdir(parents=True,exist_ok=True); os.link(p,q)
    files.append(dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)))
prefixes = ('previous-complete','previous-staged','run-previous','check-previous',
            'test-previous','test_previous','build_previous','observe_previous',
            'whole_previous','staged_previous','publish-previous')
for p in sorted(w.iterdir()):
    if not p.name.startswith(prefixes): continue
    if p.is_file() and p.suffix != '.md': copy(p)
    elif p.is_dir():
        for child in sorted(p.rglob('*')):
            if child.is_file(): copy(child)
manifest = dict(baseline=baseline,
    scope='Two private fresh complete-previous-mesh certificate variants remain unadopted. No production, public-test, standalone API or README changes; no broad adoption checks claimed. Includes the observer and initial failed offline-check invocation.',
    runtime_source_sha256={str(p):sha(p) for p in sorted(Path('warpower').glob('*.py'))},
    shared_inputs=shared, reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance']}, files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in files: assert sha(root/row['path']) == row['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes')
