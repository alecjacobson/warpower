"""Publish qualified selective step history, including rejected experiments and source provenance."""
import gzip,hashlib,json,os,subprocess,tarfile
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/local/selective-step-history';assert not out.exists()
read=lambda n:json.loads((w/n).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read('selective-step-history-production-source-v1.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==s['runtime']
for p,h in s['source_sha256'].items():assert sha(root/p)==h
assert s['unchanged_files']==265 and len(s['source_sha256'])==269
assert s['scalar_functions_ast_identical'] and s['private_policy_ast_identical'] and s['complete_mesher_only']
tests=read('selective-step-history-production-tests-v1.json');assert len(tests)==2 and all(r['exit_code']==0 and r['passed'] for r in tests)
assert '1407 passed' in (w/'selective-step-history-production-full-suite-v1.log').read_text()
commands=read('selective-step-history-final-v1/commands.json');assert len(commands)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
proof=read('selective-step-history-final-v1-independent.json');assert proof['complete'] and len(proof['cases'])==11 and len(proof['standalone'])==5
for name,r in proof['cases'].items():
 assert r['timing']['additional_pool_bytes']==44
 for phase in ['before','after']:
  p=r['independent'][phase];assert p['full_hull']['valid'] and p['boundary_self_intersection_pairs']==0 and p['all_boundary_vertex_links_single_cycles']
for r in proof['standalone'].values():assert all(r['unchanged'].values()) and r['relative_change']<.01
qualified=read('selective-step-history-qualified-v1.json');assert len(qualified)==6 and all(r['exit_code']==0 for r in qualified)
memory=read('selective-step-history-memory-v1-commands.json');assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
assert read('selective-step-history-feature-fit-v1.json')['complete']
for stem in ['pilot-v1-independent-v2','pilot-v2-independent-v3','gallery-v2-independent-v2']:
 assert read('selective-step-history-'+stem+'.json')['complete']
prefixes=('selective-step-history-','selective_step_history_','build-selective-step-history-','run-selective-step-history-','check-selective-step-history-','test_selective_step_history_','test-selective-step-history-','integrate-selective-step-history-','audit-selective-step-history-','prepare-selective-step-history-','preserve-selective-step-history-','profile-selective-step-history-','publish-selective-step-history-','verify-selective-step-history-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix!='.md' and not any(t in p.name for t in ['premerge-files','merge-audit','publication-review','git-blobs','postmerge','prior-evidence'])]
folders=['selective-step-history-'+f for f in ['pilot-v1','pilot-v2','gallery-v2','integration-v1-source','final-v1','production-loop-v1','loop-native-v1-frames','memory-v1-loop-padded-frames','memory-v1-loop-lifetime-frames']]
for f in folders:files.extend(p for p in (w/f).rglob('*') if p.is_file())
shared=[w/n for n in ['test_tangential_production_loop_v2.py','large-reuse-isolated-test.py','check-large-reuse-proof-pairs.py','check-surface-bound-reuse-final-v2.py','check-tangential-production-memory-v3.py','check-surface-query-order-final-v1.py','check-surface-push-pairs-v1.py','surface-push-production-source-v1.json','surface-push-final-v1-independent.json','surface-push-final-v1/commands.json','surface-query-order-final-v1/commands.json','tangential-production-loop-v2-independent.json','scanned-armadillo-input/armadillo.off']]
shared.extend(w/'tangential-production-loop-v2'/f for f in ['result.json','frame0.npz','frame1.npz','frame4.npz'])
for name in proof['cases']:
 for phase in ['before','after']:
  p=w/'surface-push-final-v1'/(name+'-'+phase);shared.append(p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz'))
for entry in read('selective-step-history-qualification-preparation-v1.json')['files']:
 p=Path(entry['source'])
 if p not in files:shared.append(p)
shared.extend(root/p for p in ['results/validation/vtm/surface-push-pruning/manifest.json','results/validation/vtm/step-observation/manifest.json','results/validation/vtm/tangential-topology-recovery/manifest.json'])
meta=dict(scope=__doc__,baseline=s['baseline'],runtime=s['runtime'],runtime_source_sha256=s['source_sha256'],files=[],shared_references=[],limits=['This changes trial order and may change trajectories; every proposal keeps original geometry/objective/quality guards.','The first unqualified v1 regressed uniform8K by4.98%; retained and rejected.','Sampled distance metrics are not Hausdorff bounds.','The comparable-quality wide CPU speedup goal is still unproven.','Scan input retains Stanford/libigl provenance, not MIT relicensing.'],reproduction=['NVIDIA L40, Warp1.19.0+cu12, Python3.10. Restore .work/vtm paths or adapt copies to fresh output destinations.','Use .venv/bin/python with PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process, no heavy CPU during timing.','Follow prototype contracts/pilots/checkers, then production source audit/full suite,32clean comparisons,3native traces,4isolated memory audits and final frame/feature checks.','CPU validation uses CUDA_VISIBLE_DEVICES empty,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1. Restore shared proof inputs and native validators from benchmarks/cpu/README.md.','The first-integration source snapshot is recovered byte-for-byte against the hashes recorded with its first57-test run; final60-test source is the committed runtime.'])
out.mkdir(parents=True)
for p in sorted(set(files)):
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();os.link(p,target);p.chmod(0o444)
 meta['files'].append(dict(original=str(p.relative_to(root)),path=str(target.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
for p in sorted(set(shared)):meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
# This archive is reproducible packaging: every unique source/evidence file
# stays under persistent /home/horde. Only the disposable archive copy is cached.
tag='evidence-selective-step-history-'+s['runtime'][:7]
archive_name='selective-step-history-evidence.tar.gz'
cache=Path('/opt/warpower-generated-cache/release-packaging')/tag
cache.mkdir(parents=True,exist_ok=True);archive=cache/archive_name;assert not archive.exists()
with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT,dereference=False) as tar:
    for row in meta['files']:
        for name in [row['original'],row['path']]:
            path=root/name;info=tar.gettarinfo(str(path),arcname=name)
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o444
            if info.isfile():
                with path.open('rb') as f:tar.addfile(info,f)
            else:
                assert info.islnk() and not Path(info.linkname).is_absolute()
                tar.addfile(info)
meta['release_bundle']=dict(tag=tag,name=archive_name,sha256=sha(archive),bytes=archive.stat().st_size,
    url='https://github.com/alecjacobson/warpower/releases/download/'+tag+'/'+archive_name,
    packaging_cache=str(archive),unique_evidence_remains_persistent=True)
meta['storage_policy']='Only this manifest, verification result and compact summary are committed. Raw files are downloadable release artifacts, materialized under ignored results/local and .work paths.'
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
public=root/'results/validation/vtm/selective-step-history';assert not public.exists();public.mkdir(parents=True)
os.link(p,public/'manifest.json')

print(dict(artifacts=len(meta['files']),shared=len(meta['shared_references']),bytes=sum(r['bytes'] for r in meta['files']),manifest_sha256=sha(p)))
