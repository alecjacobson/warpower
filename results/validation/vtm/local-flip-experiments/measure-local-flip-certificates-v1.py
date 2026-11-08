"""Alternating existing complete certificates on fixed retained point sets."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import warp as wp
from warpower._canonical import canonicalize
from warpower._site_bvh import SiteHierarchy
from warpower._delaunay_certificate import DelaunayCertificate
from benchmarks.run import cpu_quota_snapshot as snapshot,cpu_quota_delta as delta
wp.init();wp.set_device('cuda:0');w=Path('.work/vtm');dest=w/'local-flip-certificate-timings-v1.json';assert not dest.exists();report=dict(scope=__doc__,cases={})
for case in ['hand7','torus1000','torus4096']:
 path=w/'local-flip-observer-v2'/case/'previous-0-snapshot-0.npz'
 with np.load(path) as d:p=wp.array(d['points'],dtype=wp.vec3);t=wp.array(d['tetrahedra'],dtype=int)
 count=wp.full(1,len(t),dtype=int);tree=SiteHierarchy(p);owners,buffers=canonicalize(p,wp.zeros(len(p),dtype=float));tree.tree.owners=owners;objects=[]
 for parallel in [False,True]:
  c=DelaunayCertificate(p,t,count,tree.tree,point_parallel=parallel)
  with wp.ScopedCapture() as capture:c.run()
  wp.capture_launch(capture.graph);wp.synchronize();assert c.error.numpy()[0]==0
  objects.append((c,capture.graph))
 times=[[],[]];quota=snapshot()
 for i in range(10):
  for j in ([0,1] if i%2==0 else [1,0]):
   a=wp.Event(enable_timing=True);b=wp.Event(enable_timing=True);wp.record_event(a);wp.capture_launch(objects[j][1]);wp.record_event(b);wp.synchronize_event(b);times[j].append(wp.get_event_elapsed_time(a,b))
 q=delta(quota,snapshot());record=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu_quota_delta=q,variants=[])
 for j,(c,g) in enumerate(objects):record['variants'].append(dict(point_parallel=bool(j),samples_ms=times[j],median_ms=float(np.median(times[j])),error=int(c.error.numpy()[0]),count=int(c.tetrahedra_count.numpy()[0]),statistics=c.statistics.numpy().tolist(),local=int(c._local.numpy()[0])))
 report['cases'][case]=record;dest.write_text(json.dumps(report,indent=2)+'\n');print(case,record,flush=True)
