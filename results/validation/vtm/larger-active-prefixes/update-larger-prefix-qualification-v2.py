"""Add the existing Armadillo failures as explicit rejection guards before any final run."""
import ast
from pathlib import Path
w=Path('.work/vtm');p=w/'run-larger-prefix-final-v1.py';s=p.read_text()
old="    assert row['valid'] in [None,1] and code==0 and row['error'] in [None,0],row"
assert s.count(old)==1
s=s.replace(old,"    if 'armadillo' in name and row['valid']==0:\n        assert code==1 and row['error']==64 and not any(d['counts']),row\n    else:\n        assert row['valid'] in [None,1] and code==0 and row['error'] in [None,0],row")
needle="for name in ['delaunay1000'";idx=s.index(needle)
addition='''# Retain known rejection behavior; these are not successful-mesh timings.
for file,name in [('extended-reuse-final-v1/commands.json','uniform-armadillo8192'),('extended-reuse-gallery-dense-v1/commands.json','armadillo8192-graded-q2')]:
    refs=json.loads((w/file).read_text());ref=next(r for r in refs if r['name']==name and r['phase']=='before')
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:]
        if cmd[1]!='-m':cmd[1:2]=['-m','examples.vtm_gallery']
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase));run(name,phase,cmd,cwd)
'''
s=s[:idx]+addition+s[idx:];s=s.replace('assert len(rows)==30','assert len(rows)==34');p.write_text(s);ast.parse(s)
s=(w/'check-larger-prefix-pilot-v1.py').read_text().replace('private larger-prefix pairs','production larger-prefix pairs and retained rejection guards').replace('larger-prefix-pilot-v1','larger-prefix-final-v1').replace('assert len(commands)==8','assert len(commands)==34')
old="['reserved16384-target4096','torus8192-progressive','torus16384-fixed32','graded-torus16384']"
new="['torus16384-fixed32','torus8192-progressive','reserved16384-target4096','graded-torus16384','torus4096','torus1000','torus8192-fixed32','reserved8192-target4096','reserved16384-target8192','uniform-armadillo8192','armadillo8192-graded-q2']"
assert s.count(old)==1;s=s.replace(old,new)
# Routing changes deliberately; physical work comparisons already exclude routing counters.
p=w/'check-larger-prefix-final-v1.py';assert not p.exists();p.write_text(s);ast.parse(s)
print('Prepared 34 final jobs: nine successful pairs, two known Armadillo rejection pairs, five standalone pairs and the retained-grid pair. Not executed.')
