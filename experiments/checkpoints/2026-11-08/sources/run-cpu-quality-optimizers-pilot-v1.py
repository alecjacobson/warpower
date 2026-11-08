"""Clean repeated native CGAL quality controls while the GPU is idle; no speedup inferred."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
w=Path('.work/vtm');out=w/'cpu-quality-optimizers-pilot-v1';assert not out.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert json.loads((w/'cpu-quality-optimizers-build-v1.json').read_text())['complete']
assert json.loads((w/'cpu-quality-optimizers-contracts-v2/result.json').read_text())['complete']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=w/'cpu_mesher_quality_optimizers_v1.py';surface=w/'scanned-armadillo-input/armadillo.off'
hashes={str(p):sha(p) for p in [Path(__file__),source,w/'cpu-mesh3-quality-optimizers-v1.cpp',w/'cpu-mesh3-quality-optimizers-v1-sequential',w/'cpu-mesh3-quality-optimizers-v1-parallel',surface]}
out.mkdir();(out/'source.json').write_text(json.dumps(dict(scope=__doc__,source_sha256=hashes),indent=2)+'\n');rows=[]
for name,mode,threads,size,odt in [('sequential-coarse-odt0','sequential',1,.045,0),('sequential-finer-odt10','sequential',1,.04,10),('parallel-coarse-odt0','parallel',4,.045,0),('parallel-finer-odt10','parallel',4,.04,10)]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 cmd=[str(Path('.venv/bin/python').absolute()),str(source),'--binary',str(w/('cpu-mesh3-quality-optimizers-v1-'+mode)),'--surface',str(surface),'--size',str(size),'--facet-distance','.01','--odt-iterations',str(odt),'--feature-angle','120','--topology','manifold','--threads',str(threads),'--repeats','3','--perturb','--exude','--perturb-time-limit','5','--exude-time-limit','5','--perturb-sliver-bound','10','--exude-sliver-bound','10','--distance-binary',str(w/'surface-distance'),'--output',str(out/name)]
 start=time.time()
 with (out/(name+'.log')).open('w') as stream:
  p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='.:.work/vtm'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=900)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=900,gpu_idle_before=True)
 result=out/name/'result.json'
 if result.exists():
  d=json.loads(result.read_text());row.update(complete=d['complete'],valid=d['valid'],successful_samples=d['successful_samples'],unthrottled_samples=d['unthrottled_samples'],median_complete_seconds=d['median_unthrottled_complete_seconds'],median_mesh_and_optimization_seconds=d['median_unthrottled_mesh_and_optimization_seconds'])
 rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
 # Failed geometry is evidence, not a reason to drop the configuration.
 assert code in [0,1] and row.get('complete'),row
 assert all(sha(Path(p))==h for p,h in hashes.items())
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
(out/'complete.json').write_text(json.dumps(dict(complete=True,configurations=len(rows),native_samples=12,source_sha256=hashes),indent=2)+'\n')
