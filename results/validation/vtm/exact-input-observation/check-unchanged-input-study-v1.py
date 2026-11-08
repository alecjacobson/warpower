"""Verify observation does not change geometry/work and count bitwise equality per fresh solve."""
import hashlib, json
from pathlib import Path
import numpy as np
w = Path('.work/vtm')
helper = (w / 'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(helper, str(w / 'check-surface-bound-reuse-final-v2.py'), 'exec'))
out = w / 'unchanged-input-study-v1'; dest = w / 'unchanged-input-study-v1-independent.json'
assert not dest.exists()
jobs = json.loads((out / 'commands.json').read_text()); assert len(jobs) == 8 and all(r['exit_code'] == 0 for r in jobs)
previous = json.loads((w / 'larger-prefix-final-v1-independent.json').read_text()); assert previous['complete']
report = dict(scope=__doc__, cases={}, performance_claim=False)
def paths(stem):
    return (stem/'result.json', stem/'mesh.npz', stem/'mesh.vtu.gz') if stem.is_dir() else (stem.with_suffix('.json'), stem.with_suffix('.npz'), None)
def work(d):
    return {k:d[k] for k in ['counts','rounds','topology','refinement_rejections','quality_rejections']} | dict(steps=d.get('steps',d.get('optimizer_steps')),moves=d.get('moves',d.get('quality_moves')),routing=d.get('reconstruction_routing'))
for job in jobs:
    if not job['observed']: continue
    name = job['name']; control = out/'thin-hand-control' if name == 'thin-hand' else w/'larger-prefix-final-v1'/(name+'-after')
    ap, an, av = paths(control); bp, bn, bv = paths(out/(name+'-observed'))
    a,b = [json.loads(p.read_text()) for p in [ap,bp]]
    params = lambda d:{k:v for k,v in d['parameters'].items() if k not in ['repeats','output','save_mesh','dot']}
    assert params(a) == params(b), name
    equal_work = {k:work(a)[k] == work(b)[k] for k in work(a)}; assert all(equal_work.values()),(name,equal_work)
    with np.load(an) as x, np.load(bn) as y:
        equality = compare(x,y)
        assert all(equality[k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal']),(name,equality)
        assert all(np.array_equal(x[k],y[k]) for k in ['input_points','input_faces'])
    info = json.loads((out/(name+'-observed-observation.json')).read_text())
    routes = []
    for r in info['instances']:
        records = r['records']; assert len(records) == r['calls']
        if not records:
            routes.append(dict(capacity=r['capacity'],calls_per_solve=0,unchanged_per_solve=0)); continue
        groups = [[v[1:] for v in records if v[0] == epoch] for epoch in [1,2,3]]
        assert len(records) == sum(map(len,groups)) and groups[0] == groups[1] == groups[2], (name,r['capacity'])
        assert groups[0][0][0] == 0
        assert all(v[4] == 0 for v in groups[0]), (name,r['capacity'])
        hits = sum(v[0] == 1 and v[1] == 0 for v in groups[0])
        routes.append(dict(capacity=r['capacity'],calls_per_solve=len(groups[0]),unchanged_per_solve=hits))
    row = dict(parameters_equal=True,geometry=equality,work_equal=equal_work,routes=routes,
               calls_per_solve=sum(r['calls_per_solve'] for r in routes),unchanged_per_solve=sum(r['unchanged_per_solve'] for r in routes),
               observed_sha256=hashlib.sha256(bn.read_bytes()).hexdigest(),control_sha256=hashlib.sha256(an.read_bytes()).hexdigest())
    if name == 'thin-hand':
        row['independent'] = validate(bn,bv)
    else:
        proof = previous['cases'][name]['independent_after']
        assert proof['mesh_sha256'] == row['control_sha256'] and proof['full_hull']['valid']
        row['validity_evidence'] = dict(path=str(w/'larger-prefix-final-v1-independent.json'),case=name,scope='Exact points, canonical cells, oriented boundary, roles and input match the independently validated current control.')
    report['cases'][name] = row
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(name,row['unchanged_per_solve'],row['calls_per_solve'],routes,flush=True)
report['complete'] = True;dest.write_text(json.dumps(report,indent=2)+'\n')
