"""Predeclared ABBA repeat of the unchanged standalone hand path after a noisy pair."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'prepared-new-cell-hand-repeat-v1';assert not out.exists();primary=json.loads((w/'prepared-new-cell-final-v2/commands.json').read_text());assert len(primary)==26 and all(x['exit_code']==0 and x['quota_clean'] for x in primary)
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
refs={r['phase']:r for r in primary if r['name']=='delaunay-fitted-hand'};out.mkdir();rows=[]
for pair,phase in [(0,'before'),(0,'after'),(1,'after'),(1,'before')]:
 ref=refs[phase];cmd=ref['command'][:];dest=out/f'pair{pair}-{phase}.json';cmd[cmd.index('--output')+1]=str(dest);start=time.time()
 with dest.with_suffix('.log').open('w') as f:p=subprocess.run(cmd,cwd=ref['cwd'],env=dict(os.environ,PYTHONPATH='.'),stdout=f,stderr=subprocess.STDOUT,timeout=300)
 d=json.loads(dest.read_text());clean=all(q['throttled_periods']==0 for key in ['full','base','auxiliary','certificate'] for q in d[key]['cpu'].values());row=dict(pair=pair,phase=phase,command=cmd,cwd=ref['cwd'],source=ref['source'],exit_code=p.returncode,wall_seconds=time.time()-start,median_ms=d['full']['median_ms'],quota_clean=clean,result=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest());rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert p.returncode==0 and d['error']==0 and clean
skip={'first_replay_ms','full','base','auxiliary','certificate'}
reference=json.loads(Path(refs['before']['result']).read_text());equal=[]
for r in rows:
 d=json.loads(Path(r['result']).read_text());match={k:d[k]==reference[k] for k in reference if k not in skip};assert all(match.values());equal.append(dict(pair=r['pair'],phase=r['phase'],diagnostics_equal=match))
comparisons=[]
for pair in [0,1]:
 a=next(r for r in rows if r['pair']==pair and r['phase']=='before');b=next(r for r in rows if r['pair']==pair and r['phase']=='after');comparisons.append(dict(pair=pair,before_ms=a['median_ms'],after_ms=b['median_ms'],relative_change=b['median_ms']/a['median_ms']-1))
report=dict(scope=__doc__,reason='Primary hand pair full median +2.23% with a bimodal candidate sample set, despite unchanged runtime sources and matching diagnostic counts. Original primary samples are retained.',order=['before','after','after','before'],comparisons=comparisons,diagnostics=equal,complete=True)
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(comparisons,flush=True)
