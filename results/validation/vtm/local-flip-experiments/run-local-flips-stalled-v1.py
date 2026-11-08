"""Retain final unaccepted private candidates for stalled hand diagnosis; no speed claims."""
import hashlib,json,sys,time
from itertools import product
from pathlib import Path
import numpy as np
import warp as wp
from warpower._canonical import canonicalize
from warpower._site_bvh import SiteHierarchy
from local_flips_v1 import Candidate
wp.init();wp.set_device('cuda:0')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);dest=out/'results.json';assert not dest.exists()
report=dict(scope=__doc__,runtime='74ae277572d0a6e07cfc1442760f45b9178eee87',source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/local_flips_v1.py')]},cases={})
def positive(points,tets):
 tets=np.asarray(tets,dtype=np.int32).copy();p=points[tets].astype(np.float64);s=np.linalg.det(p[:,1:]-p[:,:1]);assert np.all(s!=0)
 for i in np.flatnonzero(s<0):tets[i,[0,1]]=tets[i,[1,0]]
 return tets

def canonical(a):return sorted(map(tuple,np.sort(a,axis=1)))
def boundary(a):
 faces={}
 for t in a:
  for k in range(4):
   f=tuple(sorted(np.delete(t,k)));faces[f]=faces.get(f,0)+1
 return sorted(k for k,v in faces.items() if v==1)

def trial(name,points,tets,capacity=None,rounds=16,expected=None,reference=None,input_hash=None):
 points=np.asarray(points,np.float32);tets=np.asarray(tets,np.int32);capacity=max(len(tets)+32,64) if capacity is None else capacity
 p=wp.array(points,dtype=wp.vec3);initial=np.full((capacity,4),-1,np.int32);initial[:len(tets)]=tets;t=wp.array(initial,dtype=int);count=wp.full(1,len(tets),dtype=int)
 tree=SiteHierarchy(p);owners,buffers=canonicalize(p,wp.zeros(len(p),dtype=float));tree.tree.owners=owners
 c=Candidate(p,t,count,tree.tree,max_rounds=rounds)
 with wp.ScopedCapture() as capture:c.run()
 wp.capture_launch(capture.graph);wp.synchronize()
 n=int(c.tetrahedra_count.numpy()[0]);e=int(c.error.numpy()[0]);a=int(c.accepted.numpy()[0]);r=int(c.rounds.numpy()[0]);actual=c.tetrahedra.numpy()[:n].copy();hist=c.history.numpy()[:r].tolist()
 assert np.array_equal(t.numpy(),initial) and count.numpy()[0]==len(tets) and np.array_equal(p.numpy().view(np.uint32),points.view(np.uint32))
 row=dict(certificate_error=int(c.cert.error.numpy()[0]),candidate_cells=int(c.cert.tetrahedra_count.numpy()[0]),points=len(points),input_cells=len(tets),capacity=capacity,round_limit=rounds,error=e,accepted=a,output_cells=n,rounds=r,history=hist,input_unchanged=True,input_hash=input_hash)
 if a:
  assert e==0 and n>0;row['boundary_unchanged']=boundary(tets)==boundary(actual);assert row['boundary_unchanged']
  if reference is not None:row['canonical_reference_equal']=canonical(reference)==canonical(actual)
 else:assert e!=0 and n==0
 npz=out/(name+'.npz');assert not npz.exists();np.savez_compressed(npz,points=points,input_tetrahedra=tets,tetrahedra=actual,candidate_tetrahedra=c.cert.tetrahedra.numpy()[:int(c.cert.tetrahedra_count.numpy()[0])])
 row['output_sha256']=hashlib.sha256(npz.read_bytes()).hexdigest();report['cases'][name]=row;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(row),flush=True)
 if expected is not None:assert bool(a)==expected,(name,row)
 return row

for path in sorted(Path('.work/vtm/local-flip-observer-v2/hand7').glob('previous-0-snapshot-[01].npz')):
 with np.load(path) as d:
  trial(path.parent.name+'-'+path.stem,d['points'],d['previous_tets'],capacity=max(len(d['previous_tets'])+1024,len(d['tetrahedra'])+1024),reference=d['tetrahedra'],input_hash=hashlib.sha256(path.read_bytes()).hexdigest())
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
