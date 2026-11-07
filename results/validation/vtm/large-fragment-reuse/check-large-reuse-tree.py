import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from benchmarks.mesh_quality import measure
from benchmarks.surface_distance import NativeSurfaceDistance
root=Path.cwd();w=root/'.work/vtm'
old=w/'extraction-split-4096/control.npz';new=w/'large-reuse-tree4096.npz'
a,b=np.load(old),np.load(new);equal={}
for k in a.files:
 x,y=a[k],b[k]
 if k in ['tetrahedra','full_tetrahedra']:
  x,y=[np.sort(v,axis=1) for v in [x,y]];x,y=[v[np.lexsort(v.T[::-1])] for v in [x,y]]
 elif k=='boundary':
  x,y=[np.asarray([np.roll(r,-int(np.argmin(r))) for r in v]) for v in [x,y]];x,y=[v[np.lexsort(v.T[::-1])] for v in [x,y]]
 equal[k]=bool(np.array_equal(x,y))
with tempfile.TemporaryDirectory(prefix='large-reuse-check-',dir=w) as temp:
 pp,tp=Path(temp)/'points.txt',Path(temp)/'tets.txt'
 np.savetxt(pp,b['points'],fmt='%.17g',header=str(len(b['points'])),comments='')
 np.savetxt(tp,b['full_tetrahedra'],fmt='%d',header=str(len(b['full_tetrahedra'])),comments='')
 full=json.loads(subprocess.run([str(w/'check-delaunay'),str(pp),str(tp)],text=True,capture_output=True,check=True).stdout)
assert full['valid']
distance=NativeSurfaceDistance(w/'surface-distance',w/'large-reuse-distance')
quality=measure(b['points'],b['tetrahedra'],b['input_points'],b['input_faces'],distance_query=distance)
before=json.loads((w/'extraction-split-4096/control.json').read_text());after=json.loads((w/'large-reuse-tree4096.json').read_text())
work={k:before[k]==after[k] for k in ['rounds','optimizer_steps','quality_moves','refinement_rejections','quality_rejections','counts','reconstruction_routing']}
quota=all(v['throttled_periods']==0 for d in [before,after] for k in ['cpu_quota_delta','initial_replays_cpu_quota_delta'] for v in d[k].values())
report=dict(baseline_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),experiment_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),arrays_equal=equal,points_max_absolute_difference=float(np.abs(a['points'].astype(float)-b['points'].astype(float)).max()),work_equal=work,all_quotas_clean=quota,before_ms=before['median_gpu_milliseconds'],after_ms=after['median_gpu_milliseconds'],full_hull=full,quality=quality,distance_oracle=distance.metadata)
(w/'large-reuse-tree4096-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['arrays_equal','points_max_absolute_difference','work_equal','all_quotas_clean','before_ms','after_ms','full_hull']},indent=2),flush=True)
