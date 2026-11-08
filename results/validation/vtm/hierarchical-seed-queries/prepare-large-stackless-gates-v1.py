"""Prepare production qualification scripts; never alter previous evidence."""
from pathlib import Path
w=Path('.work/vtm')
def save(name,s):
 p=w/name;assert not p.exists();p.write_text(s)
s=(w/'run-extended-reuse-final-v1.py').read_text()
s=s.replace('extended-reuse-final-v1','large-stackless-final-v1').replace('extended-reuse-baseline','large-stackless-baseline').replace('extended-reuse-full-suite-v1','large-stackless-full-suite-v1').replace('1121 passed','1227 passed').replace('689fd31382c490251b953046176b15ccc35e5753','5998b1384e97c04532fd5f928feb0e09c2fa702a')
start=s.index('# Original baseline measurements');end=s.index("old=json.loads",start)
s=s[:start]+'''# Repeat the six successful pilot configurations using committed production.
controls=json.loads((w/'large-stackless-pilot-v1/commands.json').read_text())
for ref in controls:
    if ref['phase']!='before':continue
    name=ref['name']
    for phase,cwd in [('before',baseline),('after',root)]:
        cmd=ref['command'][:]
        gallery='gallery' in cmd[1]
        cmd[1:2]=['-m','examples.vtm_gallery' if gallery else 'benchmarks.mesher']
        cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase) if gallery else out/(name+'-'+phase+'.json'))
        if '--save-mesh' in cmd:cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
        run(name,phase,cmd,cwd)
'''+s[end:]
s=s.replace('assert len(rows)==25','assert len(rows)==22')
save('run-large-stackless-final-v1.py',s)
s=(w/'check-large-stackless-pilot-v1.py').read_text().replace('large-stackless-pilot-v1','large-stackless-final-v1').replace('len(commands)==12','len(commands)==22').replace('private large group traversal','production large group traversal')
save('check-large-stackless-final-v1.py',s)
s=(w/'profile-extended-reuse-production-v1.py').read_text().replace('production extended fragment reuse','production hierarchical seed traversal').replace('extended-reuse-pilot-v1/torus16384-fixed32','large-stackless-final-v1/torus16384-fixed32-after')
save('profile-large-stackless-production-v1.py',s)
s=(w/'profile-extended-reuse-loop-v1.py').read_text().replace('extended fragment reuse','hierarchical seed traversal')
save('profile-large-stackless-loop-v1.py',s)
s=(w/'run-extended-reuse-native-v1.py').read_text().replace('extended-reuse-','large-stackless-').replace('Extended VTM fragment reuse','Hierarchical VTM seed traversal')
save('run-large-stackless-native-v1.py',s)
s=(w/'check-extended-reuse-native-loop-v1.py').read_text().replace('extended-reuse-','large-stackless-')
save('check-large-stackless-native-loop-v1.py',s)
# Isolated difficult cases avoid repeating every unchanged small-capacity test.
s=(w/'run-extended-reuse-memory-v2.py').read_text().replace('extended-reuse-full-suite-v1','large-stackless-full-suite-v1').replace('1121 passed','1227 passed').replace('extended-reuse-memory-v2','large-stackless-memory-v1')
start=s.index('tests=[line.strip()');end=s.index('\ngroups=',start)
s=s[:start]+'''tests = [
 'tests/test_stackless_seeds.py::test_captured_producers_preserve_exhaustive_and_rational_masks[32-0-random4097]',
 'tests/test_stackless_seeds.py::test_captured_producers_preserve_exhaustive_and_rational_masks[256-60-torus16384]',
 'tests/test_stackless_traversal.py::test_stackless_coverage_matches_recursive_pruning[32-16385]',
 'tests/test_stackless_traversal.py::test_segment_ancestors_enclose_current_groups_and_extreme_points[256-16385]',
]
collected=(w/'extended-reuse-collection-v1.log').read_text().splitlines()
for line in collected:
 if line.startswith('tests/test_large_reuse.py::') and any(x in line for x in ['[4097]','[8192]','[16384]','[-60-16384]','[0-16384]','[60-16384]']):
  tests.append(line.strip())
assert len(tests)==12, tests
'''+s[end:]
save('run-large-stackless-memory-v1.py',s)
s=(w/'run-extended-reuse-qualification-v2.py').read_text().replace('extended-reuse-qualification-v2','large-stackless-qualification-v1').replace('extended-reuse-final-v1','large-stackless-final-v1').replace('==25','==22').replace('run-extended-reuse-native-v1','run-large-stackless-native-v1').replace('run-extended-reuse-memory-v2','run-large-stackless-memory-v1')
save('run-large-stackless-qualification-v1.py',s)
