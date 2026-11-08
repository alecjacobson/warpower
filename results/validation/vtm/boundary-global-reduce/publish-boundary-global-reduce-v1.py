"""Publish an unadopted boundary grouped-reduction probe without runtime changes."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/boundary-global-reduce';assert not(out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='971e6f2e5aa641dce95c1b4ad2124e7454f11a6a'
assert subprocess.check_output(['git','diff',source,'--','warpower','tests','benchmarks','examples','README.md'])==b''
queries=json.loads((w/'boundary-global-reduce-check-v1.json').read_text());assert queries['complete'] and len(queries['cases'])==90
commands=json.loads((w/'boundary-global-reduce-pilot-v1/commands.json').read_text());assert len(commands)==4 and all(c['exit_code']==0 and c['quota_clean'] for c in commands)
checks=json.loads((w/'boundary-global-reduce-pilot-v1-independent.json').read_text());assert len(checks['cases'])==2
for c in checks['cases'].values():
 for phase in ['before','after']:
  d=c[phase]['independent'];assert d['full_hull']['valid'] and d['all_boundary_vertex_links_single_cycles'] and d['vtu_round_trip_exact'] and d['boundary_self_intersection_pairs']==0
assert checks['complete']
assert all(all(c['work_equal'].values()) and all(c['comparison'][k] for k in ['full_tets_equal','material_tets_equal','oriented_boundary_equal']) for c in checks['cases'].values())
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(('boundary-global-reduce','boundary_global_reduce','run-boundary-global-reduce','check-boundary-global-reduce','generate-boundary-global-reduce')) and '-issue-' not in p.name]
files += [p for p in (w/'boundary-global-reduce-pilot-v1').rglob('*') if p.is_file()]
files += [Path(__file__).resolve()];files=sorted(files);assert len(files)==len(set(files))
meta=dict(scope=__doc__,runtime=source,adopted=False,device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],source_sha256={},reproduction=['Restore helpers and artifacts to original .work/vtm paths in a separate checkout of the runtime commit, replacing absolute roots as needed. Harnesses refuse overwrite; validate retained evidence separately.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. Run one GPU job at a time and no heavy CPU checks during clean timing.','Build the independent CGAL validators from benchmarks/cpu. Validate all meshes in their own coordinate frames and compare work/connectivity separately.','Run this publisher with stdout outside the collected prefixes.'],limitations=['No production code/default/performance change. The conservative sum test envelope is not a general floating-point error proof.','No complete-solve gain is measured. Hand coordinates differ by at most3.469446951953614e-18; all canonical connectivity and work match.','The candidate was rejected before production memory/native capture gates.'])
for p in files:
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True);assert not p.is_symlink()
 if not target.exists():os.link(p,target)
 target.chmod(0o444);assert target.stat().st_ino==p.stat().st_ino
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py')]:
 p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
for name in ['warpower/_site_bvh.py','warpower/_boundary.py','examples/vtm_gallery.py','benchmarks/mesh_quality.py']:meta['source_sha256'][name]=sha(root/name)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
manifest=out/'manifest.json';manifest.write_text(json.dumps(meta,indent=2)+'\n');manifest.chmod(0o444)
for r in meta['files']:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
for r in json.loads((w/'boundary-global-reduce-pilot-v1/lossless-archives.json').read_text()):
 raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'] and hashlib.sha256(gzip.decompress(raw)).hexdigest()==r['raw_sha256']
result=dict(files=len(files),manifest_sha256=sha(manifest),all_artifact_source_and_archive_hashes_verified=True,runtime_unchanged=source,adopted=False)
(w/'verified-boundary-global-reduce-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
