"""Prepare publication script, keeping both accepted and unadopted experiments."""
from pathlib import Path
w=Path('.work/vtm');s=(w/'publish-hierarchical-cache-v1.py').read_text()
s=s.replace('Publish split/cached seed-query experiments, including the rejected regression.','Publish shared new-cell geometry and the unadopted strict/exact experiment.')
s=s.replace('cached-exact-seed-queries','prepared-new-cell-queries').replace('hierarchical-cache','prepared-new-cell').replace('7c3dfc74acb4e947636e428bc0ea866f66dfdf77','a4f7215fbf99249d11f765a4902c586daa606829').replace('1237 passed','1279 passed')
a=s.index('for version in [1,2,3]:');b=s.index('source_audit=',a)
s=s[:a]+'''for version in [1,2]:
 assert read(f'prepared-new-cell-pilot-v{version}-independent.json')['complete']
assert read('prepared-new-cell-grid-v2-independent.json')['complete']
assert read('prepared-new-cell-simple-guards-v1-independent.json')['complete']
assert read('prepared-new-cell-simple-grid-retry-v1.json')['exit_code']==0
'''+s[b:]
a=s.index('prefixes=');b=s.index('for name in folders:',a)
s=s[:a]+'''prefixes=('prepared-new-cell','prepared_new_cell','run-prepared-new-cell','check-prepared-new-cell','profile-prepared-new-cell','test_prepared_new_cell','test-prepared-new-cell','build-prepared-new-cell','draft-prepared-new-cell','promote-prepared-new-cell','write-prepared-new-cell','bench-prepared-new-cell','audit-prepared-new-cell','new-cell-query')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-','-issue-','-report-draft-'])]
files += [w/'proposed_unproved_queries_v1.py']
folders=['prepared-new-cell-pilot-v1','prepared-new-cell-pilot-v2','prepared-new-cell-simple-guards-v1','prepared-new-cell-final-v1']
'''+s[b:]
s=s.replace('The split-only private v1 grid regression is retained. Only the bounded-cache v2 layout was promoted.','The private strict/exact v2 experiment adds no useful benefit over simple v1 and is not adopted. Its full evidence is retained. The simple grid harness system-Python import failure and explicit-venv retry are retained.')
s=s.replace('Private prototypes use baseline7c3dfc7. Production has no private installer. Source/kernel AST checks bind the promoted implementation to the cached prototype.','Private prototypes use baselinea4f7215. Production has no private installer. Source/kernel AST checks bind the promoted kernels to simple v1. Additional simple8k/grid candidates use retained fresh v2 controls, not new interleaved pairs.')
a=s.index("meta['experiment_mapping']={");b=s.index('\nfor p in files:',a)
s=s[:a]+'''meta['experiment_mapping']={
 'simple':dict(source='prepared_new_cell_v1.py',sha256=sha(w/'prepared_new_cell_v1.py'),integration='prepared_new_cell_integration_v1.py',integration_sha256=sha(w/'prepared_new_cell_integration_v1.py'),pilot='prepared-new-cell-pilot-v1',guards='prepared-new-cell-simple-guards-v1',adopted_kernel_layout=True),
 'split':dict(source='prepared_new_cell_v2.py',sha256=sha(w/'prepared_new_cell_v2.py'),integration='prepared_new_cell_integration_v2.py',integration_sha256=sha(w/'prepared_new_cell_integration_v2.py'),pilot='prepared-new-cell-pilot-v2',grid='prepared-new-cell-grid-v2',adopted=False),
 'production':dict(source='warpower/_unproved_queries.py',runtime=runtime,final='prepared-new-cell-final-v1')}
'''+s[b:]
old="for folder,needed in [('hierarchical-seed-queries',('large-stackless-pilot-v1/commands.json','large-stackless-final-v1/commands.json')),('extended-fragment-reuse',('extended-reuse-final-v1/commands.json','extended-reuse-collection-v1.log'))]:"
new="for folder,needed in [('cached-exact-seed-queries',('hierarchical-split-pilot-v2/commands.json','hierarchical-cache-final-v1/commands.json')),('hierarchical-seed-queries',('large-stackless-pilot-v1/commands.json','large-stackless-final-v1/commands.json')),('extended-fragment-reuse',('extended-reuse-final-v1/commands.json','extended-reuse-collection-v1.log'))]:"
assert old in s;s=s.replace(old,new)
s=s.replace("['hierarchical-split-profile-v1','prepared-new-cell-full-native-v1','prepared-new-cell-loop-native-v1']","['prepared-new-cell-profile-v1','prepared-new-cell-profile-v2','prepared-new-cell-full-native-v1','prepared-new-cell-loop-native-v1']")
p=w/'publish-prepared-new-cell-v1.py';assert not p.exists();p.write_text(s)
