"""Resume driver generation with a distinct production grid benchmark filename."""
from pathlib import Path
w=Path('.work/vtm')
for stem in ['run-hierarchical-cache-final-v1','check-hierarchical-cache-final-v1','bench-hierarchical-cache-grid-v1','check-hierarchical-cache-grid-v1','profile-hierarchical-cache-production-v1','profile-hierarchical-cache-loop-v1','run-hierarchical-cache-native-v1','check-hierarchical-cache-native-loop-v1','run-hierarchical-cache-qualification-v1']:
 source=w/(stem+'.py');name=stem.replace('hierarchical-cache','prepared-new-cell')
 if stem=='bench-hierarchical-cache-grid-v1':name='bench-prepared-new-cell-production-grid-v1'
 dest=w/(name+'.py')
 s=source.read_text().replace('hierarchical-cache','prepared-new-cell').replace('1237 passed','1279 passed').replace('7c3dfc74acb4e947636e428bc0ea866f66dfdf77','a4f7215fbf99249d11f765a4902c586daa606829').replace('Cached exact VTM seed queries','Prepared current new-cell queries').replace('production cached exact seed query pairs','production shared new-cell query pairs')
 if dest.exists():
  assert stem in ['run-hierarchical-cache-final-v1','check-hierarchical-cache-final-v1'] and dest.read_text()==s
 s=s.replace('bench-prepared-new-cell-grid-v1.py','bench-prepared-new-cell-production-grid-v1.py')
 dest.write_text(s)
s=(w/'run-hierarchical-cache-memory-v1.py').read_text().replace('hierarchical-cache','prepared-new-cell').replace('1237 passed','1279 passed');a=s.index('tests = [');b=s.index('\ncollected=',a)
s=s[:a]+'''tests = [
 'tests/test_unproved_queries.py::test_captured_geometry_errors_queries_and_proofs_match[32-0-random4097]',
 'tests/test_unproved_queries.py::test_captured_geometry_errors_queries_and_proofs_match[256-60-torus16384]',
 'tests/test_unproved_queries.py::test_captured_geometry_errors_queries_and_proofs_match[32-0-flat]',
 'tests/test_unproved_queries.py::test_captured_geometry_errors_queries_and_proofs_match[256--60-cube]',
 'tests/test_unproved_queries.py::test_completion_borrows_sufficient_scratch_or_uses_original_checker[1]',
 'tests/test_unproved_queries.py::test_completion_borrows_sufficient_scratch_or_uses_original_checker[2]',
]'''+s[b:]
p=w/'run-prepared-new-cell-memory-v1.py';assert not p.exists();p.write_text(s)
print('Generated final timing/independent/native/memory drivers; distinct production grid helper')
