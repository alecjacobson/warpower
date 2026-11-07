"""Publish existing-API schedule comparisons and every explored mesh."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/quality-frontier'
assert not out.exists();out.mkdir(parents=True)
assert not subprocess.check_output(['git','diff','9228a3e','--','warpower','benchmarks','tests','README.md'],text=True)
all_rows=[]
for source,part,count in [('quality-frontier','exploration',8),('quality-frontier-polish','polish',4),('quality-frontier-confirm','confirmation',8)]:
    rows=json.loads((w/source/'results.json').read_text())
    assert len(rows)==count
    for row in rows:
        assert row['exit_code']==0 and row['valid']==1 and row['error']==0 and row['quota_clean']
        assert row['independent']['full_hull']['valid'] and row['independent']['single_cycle_boundary_links']
        assert row['independent']['quality']['referenced_vertices']==1000
    shutil.copytree(w/source,out/part)
    shutil.copy2(w/(source+'-driver.log'),out/(part+'-driver.log'))
    all_rows.extend((part,row) for row in rows)
for name in ['run-quality-frontier.py','run-quality-frontier-polish.py','run-quality-frontier-confirm.py','quality-frontier-plan.md','quality-frontier-publication-plan.md','plot-quality-frontier.py','publish-quality-frontier.py']:
    shutil.copy2(w/name,out/name)
checks=[]
for part,row in all_rows:
    m=np.load(out/part/(row['name']+'.npz'))
    with tempfile.TemporaryDirectory(prefix='frontier-surface-',dir=w) as temp:
        p=Path(temp)/'boundary.off'
        with p.open('w') as f:
            f.write(f'OFF\n{len(m["points"])} {len(m["boundary"])} 0\n')
            np.savetxt(f,m['points'],fmt='%.17g')
            np.savetxt(f,np.column_stack((np.full(len(m['boundary']),3),m['boundary'])),fmt='%d')
        r=subprocess.run([str(w/'surface-reference'),str(p)],text=True,capture_output=True,check=True)
        # Native output: vertices faces intersections, followed by witnesses.
        words=r.stdout.splitlines()[0].split()
        assert len(words)==3 and int(words[-1])==0,(row['name'],r.stdout)
        checks.append(dict(part=part,name=row['name'],output=r.stdout,stderr=r.stderr,mesh_sha256=hashlib.sha256((out/part/(row['name']+'.npz')).read_bytes()).hexdigest()))
(out/'boundary-intersections.json').write_text(json.dumps(checks,indent=2)+'\n')
comparisons=[]
rows=json.loads((out/'confirmation/results.json').read_text())
for i in range(0,len(rows),2):
    a,b=rows[i:i+2];qa,qb=[r['independent']['quality'] for r in [a,b]]
    flags={f'radius_ratio_p{p}':qb['radius_ratio_percentiles'][p]>=qa['radius_ratio_percentiles'][p] for p in ['0','1','5','50','95']}
    flags.update(boundary_rms=qb['two_sided_area_weighted_rms']<=qa['two_sided_area_weighted_rms'],absolute_volume_error=abs(qb['relative_volume_error'])<=abs(qa['relative_volume_error']),minimum_dihedral=qb['dihedral_degrees_min']>=qa['dihedral_degrees_min'],maximum_dihedral=qb['dihedral_degrees_max']<=qa['dihedral_degrees_max'])
    comparisons.append(dict(progressive=a['name'],fixed=b['name'],time_ratio=a['median_ms']/b['median_ms'],time_reduction_percent=100*(1-b['median_ms']/a['median_ms']),nondegrading_metrics=flags,all_checked_metrics_nondegrading=all(flags.values())))
# The fresh seed-2026 pair must reproduce its exploration trajectories.
reproduction=[]
for name,part,original in [('torus-2026-progressive','exploration','control'),('torus-2026-fixed-32','polish','fixed-32')]:
    a=np.load(out/'confirmation'/(name+'.npz'));b=np.load(out/part/(original+'.npz'))
    exact={k:np.array_equal(a[k],b[k]) for k in a.files if k not in ['tetrahedra','full_tetrahedra','boundary']}
    def canonical(t):
        t=np.sort(t,axis=1);return t[np.lexsort(t.T[::-1])]
    cells={k:np.array_equal(canonical(a[k]),canonical(b[k])) for k in ['tetrahedra','full_tetrahedra']}
    def oriented(t):
        t=np.asarray([np.roll(r,-int(np.argmin(r))) for r in t]);return t[np.lexsort(t.T[::-1])]
    oriented_equal=np.array_equal(oriented(a['boundary']),oriented(b['boundary']))
    assert all(exact.values()) and all(cells.values()) and oriented_equal
    reproduction.append(dict(name=name,exact_arrays=exact,canonical_cells=cells,oriented_boundary_equal=oriented_equal))
(out/'comparison.json').write_text(json.dumps(dict(scope='Within-Warp schedule study; no CPU parity claim, no runtime changes.',pairs=comparisons,repeat_output_checks=reproduction),indent=2)+'\n')
print('published',len(all_rows),'valid independently checked runs; all boundaries have zero exact self-intersections',flush=True)
