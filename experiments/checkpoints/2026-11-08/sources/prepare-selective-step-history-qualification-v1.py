"""Prepare fresh qualification drivers without changing prior checks or evidence."""
import ast,hashlib,json
from pathlib import Path
w=Path(__file__).parent;records=[]
def write(old,new,text):
 p=w/old;q=w/new;assert not q.exists();ast.parse(text);q.write_text(text)
 records.append(dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),target=str(q),target_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
for old in ['run-surface-push-production-tests-v1.py','run-surface-push-qualification-v1.py','run-surface-push-memory-v1.py','run-surface-push-loop-native-v1.py','run-surface-push-uniform-native-v1.py','profile-surface-push-uniform-v1.py','check-surface-push-native-full-v1.py','check-surface-push-loop-v1.py','run-surface-push-pipeline-v1.py','run-surface-push-independent-v1.py','check-surface-push-qualified-v1.py']:
 new=old.replace('surface-push','selective-step-history');s=(w/old).read_text().replace('surface-push','selective-step-history')
 s=s.replace('1387','1407')
 if 'production-tests' in old:s=s.replace('committed exact nearest traversal','committed bounded selective step history')
 if 'memory-v1' in old:
  s=s.replace("['tests/test_surface_bound_reuse.py','tests/test_surface_query_cache.py','tests/test_ordered_surface_queries.py','tests/test_surface_quadrature.py'],116,600", "['tests/test_selective_step_history.py','tests/test_local_backtracking.py','tests/test_vtm_controls.py'],39,600")
  s=s.replace('Audit nearest, cache, ordering and quadrature contracts','Audit selective step history, original controls and local rollback')
 if 'profile-surface-push-uniform' in old:
  s=s.replace("Path('warpower/_surface_bvh.py')", "Path('warpower/_surface_bvh.py'),Path('warpower/_vtm.py'),Path('warpower/_selective_step_history.py')")
  s=s.replace('exact child-push pruning','bounded quality-gated step history')
 if 'run-surface-push-independent' in old:
  oldcmd="str(w/'check-selective-step-history-pairs-v1.py'),'--folder',str(w/'selective-step-history-final-v1'),'--output',str(w/'selective-step-history-final-v1-independent.json')"
  assert oldcmd in s;s=s.replace(oldcmd,"str(w/'check-selective-step-history-final-v1.py')")
 if 'check-surface-push-loop' in old:s=s.replace('exact nearest-traversal','selective step history')
 write(old,new,s)
# Production case wrapper observes only its own setup and final state.
old='run-selective-step-history-case-v2.py';s=(w/old).read_text()
s=s.replace('with private selective history installed at setup','with production selective history')
s=s.replace('import selective_step_history_v2 as candidate\n','').replace('Path(candidate.__file__),','').replace('if phase=="after":candidate.install()\n','')
s=s.replace('stats=getattr(c,"private_step_statistics",None)','history=getattr(c,"step_history",None)\n        stats=None if history is None else history.statistics')
write(old,'run-selective-step-history-production-case-v1.py',s)
# Same eleven complete inputs and five standalone controls, with fresh outputs.
old='run-surface-push-final-v1.py';s=(w/old).read_text().replace('surface-push','selective-step-history').replace('Production exact nearest-traversal','Production selective-history')
s=s.replace("runenv=dict(env,", "runenv=dict(env,WARPOWER_HISTORY_PHASE=phase,WARPOWER_HISTORY_REPORT=str(out/(label+'-history.json')),WARPOWER_HISTORY_KIND='benchmark' if name.startswith('torus') else 'gallery',")
s=s.replace("cmd=template[:]", "cmd=template[:]\n        assert cmd[1:3]==['-m','benchmarks.mesher']\n        cmd=[cmd[0],str(w/'run-selective-step-history-production-case-v1.py'),*cmd[3:]]")
s=s.replace("str(w/'run-tangential-production-gallery-v1.py')","str(w/'run-selective-step-history-production-case-v1.py')")
write(old,'run-selective-step-history-final-v1.py',s)
# Changed trajectories need fresh proof unless exact arrays match a certified mesh.
old='check-selective-step-history-pilot-v3.py';s=(w/old).read_text().replace('selective-step-history-pilot-v2','selective-step-history-final-v1').replace('selective-step-history-final-v1-independent-v3.json','selective-step-history-final-v1-independent.json').replace('len(commands)==8','len(commands)==32')
s=s.replace('assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()\n','') # starts only after all clean GPU timings, during instrumented audits
s=s.replace('dict.fromkeys(r["name"] for r in commands):','dict.fromkeys(r["name"] for r in commands if not r["name"].startswith(("delaunay","diagram"))):')
s=s.replace('prior_proof_sha256=sha(proof_path),cases={})','prior_proof_sha256=sha(proof_path),cases={},standalone={})')
oldstandalone=(w/'check-surface-push-pairs-v1.py').read_text();a=oldstandalone.index("for name in dict.fromkeys(r['name'] for r in commands if r['name'].startswith")
b=oldstandalone.index("report['complete']=True",a)
standalone=oldstandalone[a:b].replace('args.folder','folder')
s=s.replace('report.update(complete=True,wall_seconds=time.time()-start)',standalone+'\nreport.update(complete=True,wall_seconds=time.time()-start)')
write(old,'check-selective-step-history-final-v1.py',s)
out=w/'selective-step-history-qualification-preparation-v1.json';assert not out.exists();out.write_text(json.dumps(dict(scope=__doc__,files=records),indent=2)+'\n')
print('Prepared',len(records),'drivers; no experiments executed.')
