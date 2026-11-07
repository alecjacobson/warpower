"""Publish closed private evidence using verified immutable hardlinks."""
import ast
import hashlib
import json
import os
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';dest=root/'results/validation/vtm/cooperative-surface-probe'
assert not dest.exists()
files=[]
for pattern in [
    'build_cooperative_rays_v*.py','cooperative_rays_v*.py','check_cooperative_rays_v*.py',
    'profile_cooperative_rays_v*.py','cooperative-rays-check-v*.log','cooperative-rays-check-v3.json',
    'cooperative-rays-profile-v3.*','capture_cooperative_rays_v1.py','cooperative-rays-capture-v1.*',
    'build_cooperative_retained_v1.py','cooperative_retained_v1.py','profile_cooperative_retained_v1.py',
    'cooperative-retained-profile-v1.*','run_cooperative_retained_v*.py','cooperative-retained-tests-v*.log',
    'check_cooperative_retained_v1.py','cooperative-retained-check-v1.*',
    'run_cooperative_retained_pilot_v1.py','check_cooperative_retained_pilot_v1.py',
    'cooperative-retained-pilot-v1.log','cooperative-retained-pilot-v1-independent.*',
    'build_cooperative_retained_split_v1.py','cooperative_retained_split_v1.py',
    'profile_cooperative_retained_split_v1.py','cooperative-retained-split-profile-v1.*',
    'geometry_query_profile_v1.py','geometry-query-profile-v1.log','extraction_split.py',
    'check-current-selection-final.py','check-large-reuse-proof-pairs.py',
]:
    found=list(w.glob(pattern));assert found,pattern
    files.extend(found)
files += list((w/'cooperative-retained-pilot-v1').glob('*'))
files += [w/'geometry-query-profile-v1/results.json',Path(__file__).resolve()]
files=sorted(set(files))
for p in files:
    assert p.is_file() and not p.is_symlink(),p
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
for name in ['cooperative-rays-check-v3.json','cooperative-rays-profile-v3.json','cooperative-rays-capture-v1.json',
             'cooperative-retained-profile-v1.json','cooperative-retained-check-v1.json','cooperative-retained-split-profile-v1.json']:
    assert json.loads((w/name).read_text())['finished']
assert json.loads((w/'cooperative-rays-check-v3.json').read_text())['all_equal']
dest.mkdir(parents=True)
rows=[]
for p in files:
    rel=p.relative_to(w);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True)
    data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
    os.link(p,q);q.chmod(0o444)
    assert hashlib.sha256(q.read_bytes()).hexdigest()==digest
    rows.append(dict(path=str(rel),bytes=len(data),sha256=digest))
manifest=dict(runtime='305758fac0575cfb4c7ded604df02a52560905ce',scope='Private cooperative query and retained-classifier probes; no production adoption.',
              files=rows,files_count=len(rows),bytes=sum(r['bytes'] for r in rows),
              retained_failures=['cooperative-rays-check-v1.log: tile vector flattening error; corrected with preserve_type=True.',
                                 'cooperative-rays-check-v2.log: positional launch_tiled dim rejected; corrected keyword.',
                                 'cooperative-retained-tests-v1.log: 27 harness failures forwarding outputs=None; corrected default empty tuple.'],
              dependencies=dict(input_meshes='results/validation/vtm/recovery-classification/recovery-classification-final',
                                native_validation='benchmarks/cpu/check_delaunay.cpp and surface_distance.cpp; existing private compiled tools, no production dependency.'),
              limitations=['Static probes exclude setup and geometry preparation except explicitly staged variants.',
                           'Fused retained whole-solve pilot covers two torus cases only; staged retained variant has no full-solve or failure-path validation.',
                           'No new full-suite, sanitizer, native-trace, scan whole-solve, or standalone benchmark adoption gate; production unchanged.',
                           'No quality-matched CPU outperformance.'])
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['files_count'],manifest['bytes'],dest)
