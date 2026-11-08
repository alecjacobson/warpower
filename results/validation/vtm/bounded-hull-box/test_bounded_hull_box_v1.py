"""Captured maximum-corner filter checked against all eight exact rational corners."""
import itertools,json,hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import warp as wp
from bounded_hull_box_v1 import maximum_corner,bounded_inside,halfspace
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def evaluate(data:wp.array2d[wp.vec3],masks:wp.array[int],accepted:wp.array[int]):
 i=wp.tid();a,b,c,lo,hi=data[i,0],data[i,1],data[i,2],data[i,3],data[i,4]
 mask=maximum_corner(halfspace(a,b,c));masks[i]=mask;accepted[i]=int(bounded_inside(a,b,c,mask,lo,hi))

def normal(p):
 a,b,c=[[F(float(x)) for x in row] for row in p]
 u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
 return a,[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]

def main():
 w=Path('.work/vtm');dest=w/'bounded-hull-box-primitive-v1.json';assert not dest.exists()
 rng=np.random.default_rng(216);rows=[];names=[]
 def add(row,name):rows.append(np.asarray(row,np.float32));names.append(name)
 for i in range(512):
  p=rng.uniform(-2,2,(3,3));n=np.cross(p[1]-p[0],p[2]-p[0]);n/=np.linalg.norm(n)
  for side,width in [(-1,.01),(-1,1),(1,.01)]:
   center=p.mean(axis=0)+side*2*n;add([*p,center-width,center+width],['random',i,side,width])
 for exponent,axis,side,translation in itertools.product([-140,-126,-60,-16,-1,0,1,2,10,60,120],range(3),[-1,1],[-4,-1,0,1,4]):
  s=np.float32(2.0**exponent);a=np.zeros(3,np.float32);a[axis]=translation*s;b=a.copy();b[(axis+1)%3]=s;c=a.copy();c[(axis+2)%3]=side*s
  for mode in ['on','inside','outside','one_step_inside']:
   q=a.copy()
   if mode=='inside':q[axis]-=side*s
   if mode=='outside':q[axis]+=side*s
   if mode=='one_step_inside':q[axis]=np.nextafter(q[axis],np.float32(-np.inf if side>0 else np.inf))
   add([a,b,c,q,q],[exponent,axis,side,translation,mode])
 for i in range(128):
  bits=rng.integers(0,2**32,size=(5,3),dtype=np.uint32);bits[(bits&0x7f800000)==0x7f800000]&=np.uint32(0xfeffffff)
  data=bits.view(np.float32);lo=np.minimum(data[3],data[4]);hi=np.maximum(data[3],data[4]);add([*data[:3],lo,hi],['finite_bits',i])
 for i in range(16):
  p=rng.uniform(-1,1,(3,3)).astype(np.float32);add([*p,p[0],p[0]],['exact_face_vertex',i])
  add([p[0],p[1],p[1],[-1,-1,-1],[1,1,1]],['flat_face',i])
 base=np.array([[0,0,0],[1,0,0],[0,1,0],[-.1,-.1,-1],[.1,.1,-.5]],np.float32)
 for slot,axis,value in itertools.product(range(5),range(3),[np.nan,np.inf,-np.inf]):
  data=base.copy();data[slot,axis]=value;add(data,['nonfinite',slot,axis,str(value)])
 for axis in range(3):
  data=base.copy();data[3,axis]=data[4,axis]+1;add(data,['reversed',axis])
 data=base.copy();data[:3,2]=np.float32(-0.0);add(data,['signed_zero'])
 data=np.asarray(rows,np.float32);wp.init();wp.set_device('cuda:0');gpu=wp.array(data,dtype=wp.vec3);masks=wp.empty(len(data),dtype=int);accepted=wp.empty(len(data),dtype=int)
 with wp.ScopedCapture() as capture:wp.launch(evaluate,len(data),inputs=[gpu,masks,accepted])
 wp.capture_launch(capture.graph);mask=masks.numpy();ok=accepted.numpy();errors=[];oracle=[];normal_errors=[]
 for i,row in enumerate(data):
  if not np.isfinite(row).all() or np.any(row[3]>row[4]):
   oracle.append(None)
   if ok[i]:errors.append([i,'unsupported',names[i]])
   continue
  a,n=normal(row[:3]);signs=[sum(n[k]*(F(float(q[k]))-a[k]) for k in range(3)) for q in itertools.product(*zip(row[3],row[4]))]
  strict=all(x<0 for x in signs);oracle.append(strict)
  if mask[i]>=0:
   if any((n[k]<0 if mask[i]&(1<<k) else n[k]>0) for k in range(3)):normal_errors.append([i,names[i]])
  if ok[i] and not strict:errors.append([i,'false_accept',names[i]])
 np.savez_compressed(w/'bounded-hull-box-primitive-v1.npz',data=data,masks=mask,accepted=ok)
 r=dict(scope=__doc__,cases=len(data),accepted=int(ok.sum()),available_corner_masks=int((mask>=0).sum()),strict_oracle_boxes=sum(x is True for x in oracle),unsupported=sum(x is None for x in oracle),errors=errors,normal_sign_errors=normal_errors,names=names,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),w/'bounded_hull_box_v1.py']})
 dest.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['names','source_sha256']},flush=True)
 assert not errors and not normal_errors and ok.sum()>50

if __name__=='__main__':main()
