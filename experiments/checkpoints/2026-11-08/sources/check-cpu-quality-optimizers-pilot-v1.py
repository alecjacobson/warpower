"""Review every completed CPU control, including mixed valid/invalid repeated outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.cpu_mesher import read_mesh
from tests.test_extraction import independent_topology
import subprocess

w=Path('.work/vtm')
folder=w/'cpu-quality-optimizers-pilot-v1'
dest=w/'cpu-quality-optimizers-pilot-v1-review.json'
assert not dest.exists()
report=dict(scope=__doc__,cases={},complete=False,limits='Three repeats per setting, including all invalid and throttled outcomes. Every initially valid output receives vertex-link and exact self-intersection checks; all rejected controls remain explicit. This is not a quality-matched speedup comparison.')
for name in [r['name'] for r in json.loads((folder/'commands.json').read_text())]:
    path=folder/name/'result.json'
    data=json.loads(path.read_text());assert data['complete']
    records=[]
    for i,r in enumerate(data['runs']):
        mesh=folder/name/f'mesh-{i}.txt'
        assert hashlib.sha256(mesh.read_bytes()).hexdigest()==r['mesh_sha256']
        row=dict(mesh_sha256=r['mesh_sha256'],initial_quality_valid=r['quality_valid'],native_valid=r['native_valid'],cpu_throttled=r['cpu_throttled'],accepted=False)
        if r['quality_valid']:
            p,t=read_mesh(mesh);boundary,components,chi=independent_topology(p,t)
            links={}
            for face in boundary:
                for j in range(3):
                    v=int(face[j]);a,b=map(int,np.delete(face,j));g=links.setdefault(v,{});g.setdefault(a,set()).add(b);g.setdefault(b,set()).add(a)
            bad=[]
            for v,g in links.items():
                seen=set();todo=[next(iter(g))]
                while todo:
                    at=todo.pop()
                    if at not in seen:seen.add(at);todo.extend(g[at]-seen)
                if len(seen)!=len(g) or any(len(x)!=2 for x in g.values()):bad.append(v)
            boundary_path=folder/name/f'review-boundary-{i}.off'
            assert not boundary_path.exists()
            with boundary_path.open('w') as stream:
                stream.write(f'OFF\n{len(p)} {len(boundary)} 0\n');np.savetxt(stream,p,fmt='%.17g');np.savetxt(stream,np.column_stack((np.full(len(boundary),3),boundary)),fmt='%d')
            result=subprocess.run([str(w/'surface-reference'),str(boundary_path)],capture_output=True,text=True,check=True)
            intersections=int(result.stdout.splitlines()[0].split()[-1])
            (folder/name/f'review-boundary-{i}-reference.txt').write_text(result.stdout+'\n'+result.stderr)
            row.update(bad_vertex_links=bad,boundary_self_intersection_pairs=intersections,components=components,boundary_euler=chi,accepted=not bad and intersections==0 and components==1 and chi==2)
        else:
            row['rejection']=r.get('quality_error',r.get('native_error'))
        records.append(row)
    report['cases'][name]=dict(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,accepted_samples=sum(r['accepted'] for r in records),accepted_unthrottled_samples=sum(r['accepted'] and not r['cpu_throttled'] for r in records))
    dest.write_text(json.dumps(report,indent=2)+'\n')
report['complete']=True
dest.write_text(json.dumps(report,indent=2)+'\n')
print({k:(v['accepted_samples'],v['accepted_unthrottled_samples']) for k,v in report['cases'].items()})
