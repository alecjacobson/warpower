"""Diagnostic failed internal state; timing may overlap offline CPU checks and is not performance evidence."""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from examples import vtm_gallery as gallery
from examples.vtm_inputs import read_off

import warp as wp
from warpower import _exterior_band as band
old_init=band.ExteriorBandRecovery.__init__
old_restore=band.ExteriorBandRecovery._restore

def observed_init(self,*args,**kwargs):
    old_init(self,*args,**kwargs)
    e=self.extraction
    self._observed={name:wp.zeros_like(value) for name,value in dict(error=e.error,diagnostics=e.diagnostics,volume=e.volume,before_volume=self.before_volume,rejected=self.rejected,removed=self.removed,rounds=self.rounds,selected=e.selected,before=self.before).items()}

def observed_restore(self):
    e=self.extraction
    values=dict(error=e.error,diagnostics=e.diagnostics,volume=e.volume,before_volume=self.before_volume,rejected=self.rejected,removed=self.removed,rounds=self.rounds,selected=e.selected,before=self.before)
    for name,value in values.items():wp.copy(self._observed[name],value)
    old_restore(self)

band.ExteriorBandRecovery.__init__=observed_init
band.ExteriorBandRecovery._restore=observed_restore

w=Path('.work/vtm');directory=w/'graded-armadillo-8192-transaction-v1';holder={};constructor=gallery.VariationalMeshingWorkspace

def keep(*args,**kwargs):
    mesh=constructor(*args,**kwargs);holder['mesh']=mesh;return mesh

gallery.VariationalMeshingWorkspace=keep
sys.argv=['capture-graded-armadillo','--case','graded-surface','--input',str(w/'scanned-armadillo-input/armadillo.off'),'--n','8192','--initial','256','--subdivisions','1','--crease-subdivisions','1','--feature-angle','120','--corner-angle','45','--repeats','0','--output',str(directory)]
try:gallery.main()
except SystemExit as exc:
    assert 'Meshing failed' in str(exc),exc
else:raise AssertionError('Expected graded extraction failure; inspect a changed trajectory')
m=holder['mesh'];e=m.extraction
assert m.valid.numpy()[0]==0 and m.error.numpy()[0]==64 and np.all(m.counts.numpy()==0)
n=int(m.active_count.numpy()[0]);nf=int(m.core.full_tetrahedra_count.numpy()[0]);assert n==8192 and nf>0
points=m.points.numpy()[:n];full=m.full_tetrahedra.numpy()[:nf];selected=e.selected.numpy()[:nf];material=full[selected!=0]
faces={};local=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]
for t in material:
    for ids in local:
        f=tuple(t[ids]);key=tuple(sorted(f));faces.setdefault(key,[]).append(f)
assert all(len(v)<=2 for v in faces.values())
boundary=np.array([v[0] for v in faces.values() if len(v)==1],np.int32)
edges=Counter(tuple(sorted((int(f[a]),int(f[b])))) for f in boundary for a,b in [(0,1),(1,2),(2,0)])
bad=[dict(vertices=list(k),incidence=v) for k,v in edges.items() if v!=2]
p0,f0=read_off(w/'scanned-armadillo-input/armadillo.off')
np.savez_compressed(directory/'failed-candidate.npz',points=points,full_tetrahedra=full,tetrahedra=material,boundary=boundary,selected=selected,circumcenters=e.circumcenters.numpy()[:nf],signed_distance=e.signed_distance.numpy()[:nf],circumradii=e.circumradii.numpy()[:nf],roles=m.core.boundary.roles.numpy()[:n],targets=m.core.boundary.targets.numpy()[:n],input_points=p0,input_faces=f0)
record=dict(scope='Diagnostic failed internal candidate only, not a valid published material mesh. Public counts remain zero.',active_sites=n,full_tetrahedra=nf,selected_tetrahedra=len(material),boundary_triangles=len(boundary),boundary_edge_incidence_histogram=dict(Counter(edges.values())),bad_edges=bad,core_error=int(m.core.error.numpy()[0]),extraction_error=int(e.error.numpy()[0]),extraction_diagnostics=int(e.diagnostics.numpy()[0]),extraction_reason=int(m._state.extraction_reason.numpy()[0]),public_error=int(m.error.numpy()[0]),public_counts=m.counts.numpy().tolist())
(directory/'failure-analysis.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

observed={name:value.numpy() for name,value in e._exterior._observed.items()}
np.savez_compressed(directory/'last-rejected-band.npz',**observed)
record={name:value.tolist() for name,value in observed.items() if name not in ['selected','before']}
record['scope']='Last rejected exterior-band trial captured before restoration. Diagnostic only; no guards or decisions changed.'
record['removed_rows']=np.flatnonzero(observed['before'] & ~observed['selected']).tolist()
(directory/'last-rejected-band.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
