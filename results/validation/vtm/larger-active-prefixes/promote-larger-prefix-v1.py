"""Promote qualified setup-only larger-prefix policy and regression coverage."""
import ast,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='df74493c97633324147cb39ef4c41a7d34c7b90a'
assert not subprocess.check_output(['git','status','--porcelain'])
assert json.loads((w/'larger-prefix-pilot-v1-independent.json').read_text())['complete']
assert json.loads((w/'larger-prefix-edge-tests-v1.json').read_text())['exit_code']==0
p=root/'warpower/_active_delaunay.py';s=p.read_text();assert s.count('n > 2048')==1 and s.count('in [4, 2048]')==1
s=s.replace('n > 2048','n > 8192').replace('in [4, 2048]','in [4, 8192]');p.write_text(s)
p=root/'warpower/_vtm.py';s=p.read_text();old='            mesh_options = dict(active_count=active_count)';assert s.count(old)==1
new='''            # Device-controlled solves can grow or shrink without reallocating.
            # Reserve fitting children during setup; each retains its complete
            # current-input certificate and the full-constructor fallback.
            mesh_options = dict(
                active_count=active_count,
                prefix_capacities=tuple(
                    n for n in (1024, 2048, 4096, 8192) if n < len(mesh_points)
                ),
            )''';s=s.replace(old,new);p.write_text(s)
s=(w/'test_larger_prefix_v1.py').read_text().replace('Captured threshold, ownership and recovery checks for larger private prefixes.','Captured threshold, ownership and recovery checks for larger VTM prefixes.')
s=s.replace('from larger_prefix_v1 import LargerPrefixDelaunayWorkspace','from warpower._active_delaunay import ActiveDelaunayWorkspace')
s=s.replace('LargerPrefixDelaunayWorkspace(', 'larger_workspace(')
where=s.index('@pytest.mark.parametrize')
s=s[:where]+'''def larger_workspace(points, **options):
    return ActiveDelaunayWorkspace(
        points,
        prefix_capacities=tuple(n for n in (1024, 2048, 4096, 8192) if n < len(points)),
        **options,
    )

'''+s[where:]
extra=(w/'test_larger_prefix_edges_v1.py').read_text();extra=extra[extra.index('@pytest.mark.parametrize'):].replace('LargerPrefixDelaunayWorkspace(', 'larger_workspace(')
s+='\n'+extra+'''

def test_oversized_explicit_child_is_rejected_before_constructor_setup():
    points=wp.zeros(16384,dtype=wp.vec3,device='cuda:0')
    active=wp.array([8193],dtype=int,device='cuda:0')
    with pytest.raises(ValueError,match=r"Prefix capacities must be in \\[4, 8192\\]"):
        ActiveDelaunayWorkspace(points,active_count=active,prefix_capacities=(8193,))
'''
p=root/'tests/test_larger_prefixes.py';assert not p.exists();p.write_text(s);ast.parse(s)
print('Promoted two setup changes and thirteen captured/guard cases. Run focused and full qualification before committing.')
