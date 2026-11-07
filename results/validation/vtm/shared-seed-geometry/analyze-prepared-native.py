"""Post-run register properties and native activity attribution, not clean timing."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

import warp as wp
from warp._src.context import get_cuda_kernel_properties
from warpower._facet_seeds import check_seeds
from warpower._prepared_seeds import prepare, check

root=Path.cwd();w=root/'.work/vtm'
properties={}
for kernel in [check_seeds,prepare,check]:
    props=get_cuda_kernel_properties(kernel,'cuda:0',block_dim=32)
    properties[kernel.key]=dict(symbol=kernel.get_mangled_name()+'_cuda_kernel_forward',**props)
db=sqlite3.connect(w/'prepared-full-native.sqlite')
sql='select s.value kernel,k.registersPerThread,k.blockX,count(*) calls,sum(k.end-k.start)/1e6 milliseconds from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value,k.registersPerThread,k.blockX order by milliseconds desc'
rows=[dict(zip(['kernel','registersPerThread','blockX','calls','milliseconds'],r)) for r in db.execute(sql)]
selected={k:[r for r in rows if r['kernel']==p['symbol']] for k,p in properties.items()}
assert len(selected['prepare'])==len(selected['check'])==1, selected
assert selected['prepare'][0]['calls']==selected['check'][0]['calls']==420,selected
with tempfile.TemporaryDirectory(prefix='prepared-baseline-',dir=w) as temp:
    path=Path(temp)/'baseline.sqlite'
    path.write_bytes(gzip.decompress((root/'results/validation/vtm/boundary-launch/native/boundary-full-native.sqlite.gz').read_bytes()))
    with sqlite3.connect(path) as old:
        old_rows=[dict(zip(['kernel','registersPerThread','blockX','calls','milliseconds'],r)) for r in old.execute(sql)]
old_selected=[r for r in old_rows if r['kernel']==properties['check_seeds']['symbol']]
assert len(old_selected)==1 and old_selected[0]['calls']==420,old_selected
report=dict(scope='Native kernel activity durations across three complete torus1000 solves; diagnostic, not clean replay medians.',baseline=old_selected,prepared=selected,properties=properties,kernels=rows)
(w/'prepared-full-native-kernel-summary.json').write_text(json.dumps(report,indent=2)+'\n')

base='a84e274425fcc4f0a85e00bbc1e2b61648888894'
old_source=subprocess.check_output(['git','show',base+':warpower/_facet_seeds.py'],text=True)
before=ast.parse(old_source);after=ast.parse((root/'warpower/_facet_seeds.py').read_text())
functions=lambda m:{n.name:ast.dump(n,include_attributes=False) for n in m.body if isinstance(n,ast.FunctionDef)}
assert functions(before)==functions(after)
changed=subprocess.check_output(['git','diff','--name-only',base,'--','warpower'],text=True).splitlines()
assert changed==['warpower/_facet_seeds.py','warpower/_prepared_seeds.py','warpower/_reuse_delaunay.py'],changed
unchanged={}
for p in sorted((root/'warpower').glob('*.py')):
    relative=str(p.relative_to(root))
    if relative in changed:continue
    old_bytes=subprocess.check_output(['git','show',base+':'+relative])
    assert p.read_bytes()==old_bytes
    unchanged[relative]=hashlib.sha256(old_bytes).hexdigest()
assert not subprocess.check_output(['git','diff',base,'--','README.md'])
isolation=dict(baseline=base,runtime='68e2386',changed=changed,unchanged_module_sha256=unchanged,original_seed_kernels_ast_equal=True,default_prepare_geometry=False,only_production_opt_in='ReuseDelaunayWorkspace._reuse_filter',readme_unchanged=True)
(w/'prepared-isolation.json').write_text(json.dumps(isolation,indent=2)+'\n')
print(json.dumps(dict(properties=properties,baseline=old_selected,prepared=selected),indent=2))
