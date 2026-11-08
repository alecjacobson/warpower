"""Publish an unadopted boundary search-seeding probe without runtime changes."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/boundary-owner-seed';assert not(out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='74601fc1670f66bf46298a3b6a78e5b91df86230'
assert subprocess.check_output(['git','diff',source,'--','warpower','tests','benchmarks','examples','README.md'])==b''
queries=json.loads((w/'boundary-owner-seed-query-check-v1.json').read_text());assert queries['complete'] and len(queries['cases'])==60
commands=json.loads((w/'boundary-owner-seed-pilot-v1/commands.json').read_text());assert len(commands)==4 and all(c['exit_code']==0 and c['quota_clean'] for c in commands)
checks=json.loads((w/'boundary-owner-seed-pilot-v1-independent.json').read_text());assert len(checks['cases'])==2
for c in checks['cases'].values():
 for phase in ['before','after']:
  d=c[phase]['independent'];assert d['full_hull']['valid'] and d['all_boundary_vertex_links_single_cycles'] and d['vtu_round_trip_exact'] and d['boundary_self_intersection_pairs']==0
assert not checks['cases']['hand7']['comparison']['material_tets_equal']
assert 'AssertionError: hand7' in (w/'boundary-owner-seed-pilot-v1-independent.log').read_text()
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(('boundary-owner-seed','boundary_owner_seed','run-boundary-owner-seed','check-boundary-owner-seed','generate-boundary-owner-seed')) and '-issue-' not in p.name]
files += [p for p in (w/'boundary-owner-seed-pilot-v1').rglob('*') if p.is_file()]
files += [Path(__file__).resolve()];files=sorted(files);assert len(files)==len(set(files))
meta=dict(scope=__doc__,runtime=source,adopted=False,device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],source_sha256={},reproduction=['Restore helpers and artifacts to original .work/vtm paths in a separate checkout of the runtime commit, replacing absolute roots as needed. Harnesses refuse overwrite; validate retained evidence separately.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. Run one GPU job at a time and no heavy CPU checks during clean timing.','Build the independent CGAL validators from benchmarks/cpu. Validate own-frame meshes separately from the intentionally retained strict connectivity failure.','Run this publisher with stdout outside the collected prefixes.'],limitations=['No production code/default/performance change. The direct query checks are not a full trajectory equivalence guarantee.','Hand7 has a measured4.25% slowdown and changed material/boundary connectivity despite independent valid geometry and matching work; no quality-equivalent speedup claim or causal-coordinate explanation.','The candidate was rejected before production memory/native capture gates.'])
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
for r in json.loads((w/'boundary-owner-seed-pilot-v1/lossless-archives.json').read_text()):
 raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'] and hashlib.sha256(gzip.decompress(raw)).hexdigest()==r['raw_sha256']
result=dict(files=len(files),manifest_sha256=sha(manifest),all_artifact_source_and_archive_hashes_verified=True,runtime_unchanged=source,adopted=False)
(w/'verified-boundary-owner-seed-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
