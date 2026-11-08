"""Exact offline identification of retained hand four-cell replacements."""
import json,itertools,hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np
w=Path('.work/vtm');dest=w/'local-flip-stalls-diagnosis-v1.json';assert not dest.exists();report=dict(scope=__doc__,cases={})
def orient(p,t):
 q=[[F(float(x)) for x in p[i]] for i in t];a,b,c=[[q[i][j]-q[0][j] for j in range(3)] for i in range(1,4)]
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def boundary(p,tets):
 faces={}
 for row in tets:
  t=list(row)
  if orient(p,t)<0:t[0],t[1]=t[1],t[0]
  for k in range(4):
   f=t[:k]+t[k+1:];s=(-1)**(k+sum(f[i]>f[j] for i in range(3) for j in range(i+1,3)));key=tuple(sorted(f));faces[key]=faces.get(key,0)+s
 return {str(k):v for k,v in faces.items() if v}
for i in range(2):
 candidate=w/'local-flips-stalled-v1'/f'hand7-previous-0-snapshot-{i}.npz';reference=w/'local-flip-observer-v2/hand7'/f'previous-0-snapshot-{i}.npz'
 with np.load(candidate) as a,np.load(reference) as b:
  p=a['points'];assert np.array_equal(p,b['points']);x=set(map(tuple,np.sort(a['candidate_tetrahedra'],axis=1).tolist()));y=set(map(tuple,np.sort(b['tetrahedra'],axis=1).tolist()));remaining=x-y;components=[]
  while remaining:
   group={remaining.pop()};changed=True
   while changed:
    add={t for t in remaining if any(len(set(t)&set(s))==3 for s in group)};changed=bool(add);group|=add;remaining-=add
   components.append(sorted(group))
  rows=[];covered=set()
  for group in components:
   verts=set().union(*map(set,group));replacement=sorted(t for t in y-x if set(t)<=verts);covered.update(replacement)
   oldedge=set.intersection(*map(set,group));newedge=set.intersection(*map(set,replacement));ring=sorted(verts-oldedge)
   oldvol=sum(abs(orient(p,t)) for t in group);newvol=sum(abs(orient(p,t)) for t in replacement);beq=boundary(p,group)==boundary(p,replacement)
   row=dict(vertices=sorted(verts),coordinates=p[sorted(verts)].tolist(),old_cells=group,new_cells=replacement,old_edge=sorted(oldedge),new_edge=sorted(newedge),ring=ring,ring_exact_determinant=str(orient(p,ring)),old_exact_six_volume=str(oldvol),new_exact_six_volume=str(newvol),oriented_boundary_equal=beq,zero_orientation_quadruples=[list(t) for t in itertools.combinations(sorted(verts),4) if orient(p,t)==0])
   assert len(group)==len(replacement)==4 and len(verts)==6 and len(oldedge)==len(newedge)==2 and oldvol==newvol and beq,row;rows.append(row)
  assert covered==y-x
  report['cases'][str(i)]=dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),groups=rows)
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
