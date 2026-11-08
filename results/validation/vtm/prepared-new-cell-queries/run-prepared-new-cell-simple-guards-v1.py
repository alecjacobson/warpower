"""Simple prototype candidates against retained fresh v2 controls, not new pairs."""
import hashlib,json,subprocess,time
from pathlib import Path
w=Path('.work/vtm').resolve();out=w/'prepared-new-cell-simple-guards-v1';assert not out.exists()
assert len(json.loads((w/'prepared-new-cell-all-timing-v2.json').read_text()))==2
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source=='a4f7215fbf99249d11f765a4902c586daa606829'
out.mkdir();refs=json.loads((w/'prepared-new-cell-pilot-v2/commands.json').read_text());rows=[]
for name in ['torus8192-progressive','torus8192-fixed32','retained-grid']:
 if name=='retained-grid':
  control=w/'prepared-new-cell-grid-v2-before.json';cmd=[str(Path('.venv/bin/python').resolve()),str(w/'bench-prepared-new-cell-grid-v1.py'),'--candidate','--output',str(out/(name+'.json'))]
 else:
  ref=next(r for r in refs if r['name']==name and r['phase']=='before');control=Path(ref['result']);cmd=ref['command'][:];cmd[1:2]=[str(w/'run-prepared-new-cell-mesher-v1.py'),'--candidate'];cmd[cmd.index('--output')+1]=str(out/(name+'.json'));cmd[cmd.index('--save-mesh')+1]=str(out/(name+'.npz'))
 start=time.time()
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 result=out/(name+'.json');d=json.loads(result.read_text());quota={k:d[k] for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta','first_replay_cpu_quota_delta','initial_cpu_quota_delta'] if k in d};clean=all(v['throttled_periods']==0 for q in quota.values() for v in q.values())
 row=dict(name=name,command=cmd,source=source,prototype_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['prepared_new_cell_v1.py','prepared_new_cell_integration_v1.py']},control=str(control),control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),control_scope='Retained fresh control from preceding v2 timing batch, not a new interleaved pair',exit_code=p.returncode,wall_seconds=time.time()-start,result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),median_ms=d.get('median_ms',d.get('median_gpu_milliseconds')),error=d['error'],quota_clean=clean)
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert p.returncode==0 and d['error']==0 and clean
