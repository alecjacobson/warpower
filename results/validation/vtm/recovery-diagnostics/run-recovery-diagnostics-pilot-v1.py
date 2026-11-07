"""Fresh clean paired scan solves; only duplicate input serialization uses cache."""
import gzip,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'recovery-diagnostics-pilot-v1';assert not out.exists();out.mkdir();rows=[]
ref=next(r for r in json.loads((w/'current-selection-final/commands.json').read_text()) if r['name']=='armadillo-graded' and r['phase']=='after')
canonical=w/'scanned-armadillo-input/armadillo.off'
export=Path('/opt/warpower-generated-cache/recovery-diagnostics-export');export.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.:.work/vtm',PYTHONDONTWRITEBYTECODE='1',WARP_CACHE_PATH='/opt/warpower-generated-cache')
for phase in ['control','candidate']:
 stem=out/phase;stem.mkdir();duplicate=export/(phase+'-v1-input.off');assert not duplicate.exists();(stem/'input.off').symlink_to(duplicate)
 cmd=ref['command'][:];cmd[cmd.index('--output')+1]=str(stem)
 if phase=='candidate':cmd[1:3]=[str(w/'run-recovery-diagnostics-gallery-v1.py')]
 start=time.time()
 with stem.with_suffix('.log').open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
 row=dict(phase=phase,command=cmd,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),exit_code=code,wall_seconds=time.time()-start,scope='Private replay pilot. Candidate kernel is preloaded before benchmark setup; setup comparison is not valid.')
 if code==0:
  d=json.loads((stem/'result.json').read_text());row.update(median_ms=d['median_additional_replay_ms'],valid=d['valid'],error=d['error'],quota_clean=all(v['throttled_periods']==0 for key in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in d[key].values()))
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0 and row['valid']==1 and row['error']==0 and row['quota_clean'],row
 assert hashlib.sha256(duplicate.read_bytes()).digest()==hashlib.sha256(canonical.read_bytes()).digest()
 (stem/'input.off').unlink();os.link(canonical,stem/'input.off')
 archive=[]
 for name in ['mesh.vtu','boundary.off']:
  p=stem/name;data=p.read_bytes();dest=p.with_suffix(p.suffix+'.gz');dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
  archive.append(dict(path=str(dest),raw_sha256=hashlib.sha256(data).hexdigest(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));p.unlink()
 (stem/'lossless-exports.json').write_text(json.dumps(archive,indent=2)+'\n')
