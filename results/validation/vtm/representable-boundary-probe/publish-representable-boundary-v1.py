"""Publish immutable, hashed experimental evidence without changing runtime."""
import hashlib,json,os
from pathlib import Path
w=Path('.work/vtm');out=Path('results/validation/vtm/representable-boundary-probe')
assert not (out/"manifest.json").exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted(p for p in w.iterdir() if p.is_file() and p.name.startswith(('representable','test_representable','run-representable','check-representable')) and p.suffix!='.md')
files+=sorted(p for p in (w/'representable-full-v1').rglob('*') if p.is_file())
files += [w/'stable-center-final-hand-final-candidate.npz',Path(__file__).relative_to(Path.cwd())]
assert len(files)==len(set(files))
for row in json.loads((w/'representable-full-v1/commands.json').read_text()):
 assert row['exit_code']==(1 if row['case']=='coarse-hand' else 0)
for log,text in [('representable-boundary-tests-v1.log','18 passed'),('representable-transaction-tests-v2.log','5 passed'),('representable-existing-tests-v1.log','44 passed')]:
 assert text in (w/log).read_text()
meta=dict(runtime='6f7efc9f1515a18077053d23f3a306c4a6cdc47a',scope='Unadopted exact-midpoint proposal, full-solve quality tradeoff and read-only selective-backtracking attribution. No runtime changes or production performance claim.',
 device='NVIDIA L40',warp='1.19.0',files=[],shared_inputs=[],shared_helpers=[],source_sha256={},
 reproduction='Restore each original path from its path entry in a separate checkout of runtime. Shared helpers and inputs retain their listed original paths. Commands in records use the original workspace path; substitute the checkout root. Native independent validators are built from benchmarks/reference; this bundle does not include executables.',
 limitations=['No new standalone/full-suite/native-profiler/sanitizer adoption gate.','Coarse hand explicitly invalid in both full pilots.','Instrumented full-solve correction counters count core-accepted moves which outer quality stages may later roll back.','Selective observer coordinates differ below 1e-8 from corresponding unobserved pilots; work and roles agree.'])
for p in files:
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True);
 if not target.exists():os.link(p,target)
 assert p.stat().st_ino==target.stat().st_ino
 target.chmod(0o444)
 assert sha(target)==sha(p)
 meta['files'].append(dict(original=str(p),path=str(target),bytes=p.stat().st_size,sha256=sha(p)))
for name,original,path in [
 ('shared_inputs',w/'surface-bound-reuse-final/thin-hand-after/mesh.npz',Path('results/validation/vtm/surface-bound-reuse/surface-bound-reuse-final/thin-hand-after/mesh.npz')),
 ('shared_inputs',w/'surface-bound-reuse-final/torus1000-after.npz',Path('results/validation/vtm/surface-bound-reuse/surface-bound-reuse-final/torus1000-after.npz')),
 ('shared_helpers',w/'check-surface-bound-reuse-final-v2.py',Path('results/validation/vtm/surface-bound-reuse/check-surface-bound-reuse-final-v2.py')),
 ('shared_helpers',w/'check-large-reuse-proof-pairs.py',Path('results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'))]:
 assert sha(original)==sha(path)
 meta[name].append(dict(original=str(original),path=str(path),sha256=sha(path)))
for p in ['warpower/_vtm.py','warpower/_selective.py','warpower/_mesher.py','examples/vtm_gallery.py','examples/vtm_inputs.py','benchmarks/feature_quality.py']:
 meta['source_sha256'][p]=sha(Path(p))
meta['reference_binaries_sha256']={p:sha(w/p) for p in ['check-delaunay','surface-distance','surface-reference']}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'manifest.json').chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=str(out/'manifest.json'))))
