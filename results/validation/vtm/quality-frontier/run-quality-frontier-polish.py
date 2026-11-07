"""Exploratory #37 complete-solve quality/time frontier; no default changes.

Run only after outstanding GPU jobs and #158 validation finish. All GPU timing
precedes the independent host geometry work. Invalid results are retained.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

root=Path.cwd();w=root/'.work/vtm';out=w/'quality-frontier-polish';python=root/'.venv/bin/python'
assert not out.exists();out.mkdir()
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','diff','--','warpower','benchmarks','examples'],text=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
configurations=[('fixed-24',1000,0,24,10,4),('fixed-28',1000,0,28,10,4),('fixed-32',1000,0,32,10,4),('fixed-40',1000,0,40,10,4)]
rows=[]
for name,initial,growth,polish,selective,quality in configurations:
    stem=out/name
    cmd=[str(python),'-m','benchmarks.mesher','--n','1000','--initial',str(initial),'--growth-iterations',str(growth),'--polish-iterations',str(polish),'--selective-iterations',str(selective),'--quality-rounds',str(quality),'--repeats','3','--save-mesh',str(stem.with_suffix('.npz')),'--output',str(stem.with_suffix('.json'))]
    start=time.time();timed_out=False
    with stem.with_suffix('.log').open('w') as log:
        try:code=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        except subprocess.TimeoutExpired:code=124;timed_out=True
    row=dict(name=name,command=cmd,source=revision,exit_code=code,timeout_seconds=180,timed_out=timed_out,wall_seconds=time.time()-start)
    if stem.with_suffix('.json').exists():
        d=json.loads(stem.with_suffix('.json').read_text())
        row.update(valid=d['valid'],error=d['error'],counts=d['counts'],median_ms=d['median_gpu_milliseconds'],reported_quality=d.get('quality'),quota_clean=all(v['throttled_periods']==0 for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[k].values()))
    elif not timed_out:
        rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n');raise RuntimeError(row)
    rows.append(row);(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,code,row.get('valid'),row.get('median_ms'),row.get('quota_clean'),flush=True)

# Offline independent geometry; no GPU benchmarking remains in this process.
import numpy as np
from benchmarks.mesh_quality import measure
from benchmarks.surface_distance import NativeSurfaceDistance

reference_input=None
for row in rows:
    if row.get('valid')!=1:continue
    stem=out/row['name'];m=np.load(stem.with_suffix('.npz'))
    input_hash=hashlib.sha256(m['input_points'].tobytes()+m['input_faces'].tobytes()).hexdigest()
    if reference_input is None:reference_input=input_hash
    assert input_hash==reference_input
    with tempfile.TemporaryDirectory(prefix='full-hull-',dir=out) as temp:
        pp,tp=Path(temp)/'points.txt',Path(temp)/'tets.txt'
        np.savetxt(pp,m['points'],fmt='%.17g',header=str(len(m['points'])),comments='')
        np.savetxt(tp,m['full_tetrahedra'],fmt='%d',header=str(len(m['full_tetrahedra'])),comments='')
        full=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout)
    assert full['valid'],row['name']
    distance=NativeSurfaceDistance(w/'surface-distance',out)
    quality=measure(m['points'],m['tetrahedra'],m['input_points'],m['input_faces'],distance_query=distance)
    row['independent']=dict(input_sha256=input_hash,mesh_sha256=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest(),full_hull=full,quality=quality,distance_oracle=distance.metadata,role_counts={str(int(k)):int(v) for k,v in zip(*np.unique(m['roles'],return_counts=True))})
    links={}
    for face in m['boundary']:
        for i in range(3):
            vertex=int(face[i]);a,b=map(int,np.delete(face,i));g=links.setdefault(vertex,{})
            g.setdefault(a,set()).add(b);g.setdefault(b,set()).add(a)
    for vertex,g in links.items():
        seen=set();todo=[next(iter(g))]
        while todo:
            at=todo.pop()
            if at not in seen:seen.add(at);todo.extend(g[at]-seen)
        assert len(seen)==len(g) and all(len(neighbors)==2 for neighbors in g.values()),(row['name'],vertex)
    row['independent']['single_cycle_boundary_links']=True
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('independently checked',row['name'],flush=True)
