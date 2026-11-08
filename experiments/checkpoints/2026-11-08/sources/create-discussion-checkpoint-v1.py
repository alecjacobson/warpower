"""Create a compact discussion checkpoint without committing raw evidence or executables."""
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'experiments/checkpoints/2026-11-08';assert not out.exists();out.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prefixes=('selective-step-history-','selective_step_history_','build-selective-step-history-','run-selective-step-history-','check-selective-step-history-','test_selective_step_history_','test-selective-step-history-','integrate-selective-step-history-','audit-selective-step-history-','prepare-selective-step-history-','preserve-selective-step-history-','profile-selective-step-history-','publish-selective-step-history-','verify-selective-step-history-','cpu-quality-optimizers-','cpu_mesher_quality_optimizers_','cpu-mesh3-quality-optimizers-','prepare-cpu-quality-optimizers-','build-cpu-quality-optimizers-','check-cpu-quality-optimizers-','run-cpu-quality-optimizers-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes)]
for p in w.iterdir():
 if p.is_dir() and p.name.startswith(('selective-step-history-','cpu-quality-optimizers-')) and not p.name.endswith('-baseline'):
  files.extend(q for q in p.rglob('*') if q.is_file())
files=sorted(set(files));scripts=[p for p in files if p.parent==w and p.suffix in ['.py','.cpp']]
scripts.append(Path(__file__).resolve());sources=[]
for p in scripts:
 target=out/'sources'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
 sources.append(dict(original=str(p.relative_to(root)),checkpoint=str(target.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size))
(out/'source-manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
(out/'raw-evidence-inventory.json').write_text(json.dumps(dict(scope='Inventory only. Raw files remain in persistent local storage; release publication is not complete.',files=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in files if p.suffix not in ['.py','.cpp','.md']]),indent=2)+'\n')
read=lambda name:json.loads((w/name).read_text())
proof=read('selective-step-history-final-v1-independent.json');cpu=read('cpu-quality-optimizers-pilot-v1-review.json');memory=read('selective-step-history-memory-v2.json')
summary=dict(runtime='b343f703f562baefbaabf83437c24ed5cf80762a',base='afec85de70d6c65b95e2c8d745edef5ecc9ca965',goal_status='Paused at user request; CPU outperformance goal remains unmet.',draft_pr=260,issues=dict(step_history=257,cpu_optimizer_baseline=261,artifact_storage=262),production_tests=1407,targeted_tests=60,clean_final_jobs=32,independent_meshes=22,geometry_complete=proof['complete'],feature_fit=read('selective-step-history-feature-fit-v1.json'),memory=dict(complete=False,original_mixed_batch_exit=124,isolated_rows=memory['rows'],user_stop=read('selective-step-history-user-stop-v1.json')),gpu_cases={n:dict(timing=r['timing'],comparison=r['comparison'],quality_comparison=r['quality_comparison'],history_statistics=r['history_statistics']) for n,r in proof['cases'].items()},standalone=proof['standalone'],cpu_cases={})
for name,review in cpu['cases'].items():
 d=read('cpu-quality-optimizers-pilot-v1/'+name+'/result.json')
 summary['cpu_cases'][name]=dict(parameters=d['parameters'],accepted_samples=review['accepted_samples'],accepted_unthrottled_samples=review['accepted_unthrottled_samples'],median_complete_seconds=d['median_unthrottled_complete_seconds'],median_mesh_and_optimization_seconds=d['median_unthrottled_mesh_and_optimization_seconds'],runs=[dict(vertices=r.get('vertices'),native_complete_seconds=r.get('complete_seconds'),cpu_throttled=r['cpu_throttled'],quality_valid=r['quality_valid'],quality=r.get('independent_quality'),quality_error=r.get('quality_error'),optimizer_return_codes={k:r.get(k) for k in ['odt_return_code','perturb_return_code','exude_return_code']}) for r in d['runs']])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['selective-step-history-production-source-v1.json','selective-step-history-production-tests-v1.json','cpu-quality-optimizers-source-v1.json','cpu-quality-optimizers-build-v1.json','cpu-quality-optimizers-checker-revision-v2.json','repository-size-inventory-v1.json']:
 shutil.copyfile(w/name,out/name)
(out/'STATUS.md').write_text('''# Discussion checkpoint — 2026-11-08

Work is paused at the user's request. No background experiments should be resumed
until the user asks to continue. PR #260 remains a draft and has not been merged.

The branch contains the selective-step-history runtime commit `b343f703` plus
this compact checkpoint. All 1,407 production tests, 60 targeted tests, the full
five-phase 8K recovery loop, 32 clean timing jobs, and independent checks of all
22 final meshes passed. The final 16K torus improves from 4.378 to 2.898 seconds
(1.51×). Standalone Delaunay timings change by −0.31% to +0.09%; power diagrams
by −3.29%, with their implementations unchanged. All three native trace jobs
completed, but final combined qualification is unfinished.

The first mixed-workspace memory audit hit its 600-second deadline. Its isolated
replacement preserves all 37 contracts. Completed groups and the user-stopped
job are recorded in `summary.json`; remaining audits and frame comparisons are
required before merging. A user-stopped job is not a passing qualification.

The optional CGAL perturbation/exudation benchmark code is a private prototype,
not integrated into the public benchmark. Both native builds and functional
contracts passed. An initial checker incorrectly expected exudation to change
the global maximum vertex weight; its corrected version checks the actual
changed weighted vertices and mesh quality and retains the failed invocation.

The repeated scan pilot has six independently valid, unthrottled coarse outputs:
CGAL sequential median 0.616 s and four-thread median 0.427 s, at roughly6,000
sites. They have stronger worst-element quality and tighter surface fit but lower
mean radius ratio than the 8K Warp output. All six finer ODT10 outputs failed
independent material topology checks and remain retained. These are not a
matched-quality CPU speedup result; the requested wide GPU advantage is unmet.

Raw experiment snapshots, profiler traces, meshes and native executables are
excluded from this checkpoint. Their paths and hashes are inventoried; originals
remain on persistent local storage. Forward publication is prepared to use
checksum-verified release bundles, with only compact manifests/summaries in Git.
No release bundle has been published and no historical Git migration performed.

The `sources/` directory preserves the current private scripts with original
`.work/vtm` path mappings in `source-manifest.json`. Some are superseded or
unexecuted preparations, not finished public APIs. They depend on retained inputs
and helpers identified by their source and the raw inventory. In particular,
release publication/checking scripts still need adaptation to the isolated memory
report; do not execute them as a completed qualification pipeline.

Next steps after discussion: finish memory/frame qualification and update the
release publication scripts before merging #260; then continue the CGAL quality/
size/fit comparison (#244/#261/#40). Keep raw evidence out of ordinary Git (#262).
''')
size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());print(dict(files=sum(1 for p in out.rglob('*') if p.is_file()),bytes=size,source_files=len(scripts)))
assert size<2_000_000
