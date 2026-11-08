"""Prepare production qualification; commands require completed suite and committed source."""
import ast
from pathlib import Path
w=Path('.work/vtm')
s=(w/'run-prepared-new-cell-final-v1.py').read_text()
s=s.replace('prepared-new-cell','larger-prefix').replace('1279 passed','1292 passed').replace('a4f7215fbf99249d11f765a4902c586daa606829','df74493c97633324147cb39ef4c41a7d34c7b90a')
start=s.index('# Repeat the six');s=s[:start]+'''# Seven original complete-solve controls, plus both newly useful reservations.
controls=json.loads((w/'prepared-new-cell-final-v2/commands.json').read_text())
names=['torus16384-fixed32','torus8192-progressive','reserved16384-target4096','graded-torus16384','torus4096','torus1000','torus8192-fixed32']
lookup={r['name']:r for r in controls if r['phase']=='after'}
for name in [*names,'reserved8192-target4096','reserved16384-target8192']:
    ref=lookup[name] if name in lookup else lookup['reserved16384-target4096']
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:];assert cmd[1]=='-m'
        gallery=cmd[2]=='examples.vtm_gallery'
        if name=='reserved8192-target4096':cmd[cmd.index('--capacity')+1]='8192'
        if name=='reserved16384-target8192':cmd[cmd.index('--n')+1]='8192'
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase) if gallery else out/(name+'-'+phase+'.json'))
        if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
        run(name,phase,cmd,cwd)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=lookup[name]['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
for phase,cwd in [('before',baseline),('after',root)]:
    cmd=[str(python),str(w/'bench-larger-prefix-production-grid-v1.py'),'--output',str(out/('retained-grid-'+phase+'.json'))]
    run('retained-grid',phase,cmd,cwd)
assert len(rows)==30
'''
s=s.replace("    if row['valid']==0:assert code==1 and row['error']==64 and not any(d['counts']),row\n    else:assert code==0 and row['error'] in [None,0],row","    assert row['valid'] in [None,1] and code==0 and row['error'] in [None,0],row")
p=w/'run-larger-prefix-final-v1.py';assert not p.exists();p.write_text(s);ast.parse(s)
s=(w/'bench-prepared-new-cell-production-grid-v1.py').read_text().replace('prepared-new-cell','larger-prefix')
# Record the two setup-policy sources as well as the unchanged geometric helper.
s=s.replace("'warpower/_proved_completion.py'","'warpower/_active_delaunay.py','warpower/_vtm.py','warpower/_proved_completion.py'")
p=w/'bench-larger-prefix-production-grid-v1.py';assert not p.exists();p.write_text(s);ast.parse(s)
s=(w/'run-prepared-new-cell-memory-v1.py').read_text().replace('prepared-new-cell','larger-prefix').replace('1279 passed','1292 passed')
a=s.index('tests = [');b=s.index('\ngroups=',a)
s=s[:a]+'''collected=(w/'larger-prefix-collection-v1.log').read_text().splitlines()
tests=[line.strip() for line in collected if line.startswith('tests/test_larger_prefixes.py::') and 'oversized_explicit' not in line]
assert len(tests)==12,tests
tests += [
 'tests/test_active_delaunay.py::test_changing_prefixes_producers_consumers_and_failures_in_one_graph',
 'tests/test_active_delaunay.py::test_vtm_rollback_restores_active_mesh_across_routed_construction',
]
assert len(tests)==14
'''+s[b:]
p=w/'run-larger-prefix-memory-v1.py';assert not p.exists();p.write_text(s);ast.parse(s)
print('Prepared 30 production timing jobs, grid helper, and 28 isolated memory jobs; none executed.')
