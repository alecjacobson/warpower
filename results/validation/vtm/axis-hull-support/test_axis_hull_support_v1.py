"""Exact rational AABB support checks for the private coordinate-plane predicate."""
import hashlib,itertools,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import warp as wp
from axis_hull_support_v1 import axis_supported
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def evaluate(data:wp.array2d[wp.vec3],output:wp.array[int]):
 i=wp.tid();output[i]=int(axis_supported(data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5]))

def orient(a,b,c,d):
 q=[[F(float(x)) for x in v] for v in [a,b,c,d]]
 u,v,w=[[q[i][k]-q[0][k] for k in range(3)] for i in range(1,4)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])

def main():
 rows=[];expected=[];names=[];valid=[]
 for exponent,translation,axis,side in itertools.product([-140,-120,-60,0,60,120],[-4,-1,0,1,4],range(3),[-1,1]):
  s=np.float32(2.0**exponent);plane=np.float32(translation*s);j=(axis+1)%3;k=(axis+2)%3
  a=np.zeros(3,np.float32);a[axis]=plane;b=a.copy();b[j]=s;c=a.copy();c[k]=s
  d=a.copy();d[j]=s*np.float32(.25);d[k]=s*np.float32(.25);d[axis]=plane+side*s
  if orient(a,b,c,d)>0:b,c=c,b
  assert orient(a,b,c,d)<0
  lo=np.minimum.reduce([a,b,c,d])-s;hi=np.maximum.reduce([a,b,c,d])+s
  if side>0:lo[axis]=plane
  else:hi[axis]=plane
  for mode in ['on','inside','outside','near_axis','flat_opposite']:
   data=np.asarray([a,b,c,d,lo,hi],np.float32).copy()
   if mode=='inside':data[4 if side>0 else 5,axis]=plane+side*s*np.float32(.25)
   elif mode=='outside':data[4 if side>0 else 5,axis]=np.nextafter(plane,np.float32(-np.inf if side>0 else np.inf))
   elif mode=='near_axis':data[1,axis]=np.nextafter(plane,np.float32(np.inf if side>0 else -np.inf))
   elif mode=='flat_opposite':data[3,axis]=plane
   rows.append(data);expected.append(mode in ['on','inside']);names.append([exponent,translation,axis,side,mode]);valid.append(orient(*data[:4])<0)
 # Signed-zero equality is geometric equality, including all supported axes.
 for axis,side in itertools.product(range(3),[-1,1]):
  index=names.index([0,0,axis,side,'on']);data=rows[index].copy();data[1,axis]=np.float32(-0.0);rows.append(data);expected.append(True);names.append(['signed_zero',axis,side]);valid.append(True)
 # Explicitly unsupported scalar preconditions must not produce a shortcut.
 base=rows[names.index([0,1,2,1,'on'])].copy()
 for label,which,value in [('nan_plane',0,np.nan),('inf_opposite',3,np.inf),('nan_lower',4,np.nan),('inf_upper',5,np.inf)]:
  data=base.copy()
  if which==0:data[:3,2]=value
  else:data[which,2]=value
  rows.append(data);expected.append(False);names.append([label]);valid.append(False)
 data=base.copy();data[4,2]=data[5,2]+1;rows.append(data);expected.append(False);names.append(['reversed_box']);valid.append(False)
 rng=np.random.default_rng(214)
 for i in range(128):
  p=rng.normal(size=(4,3)).astype(np.float32)
  if orient(*p)>0:p[[1,2]]=p[[2,1]]
  assert orient(*p)<0
  data=np.concatenate([p,[p.min(axis=0)-1,p.max(axis=0)+1]])
  rows.append(data);expected.append(False);names.append(['general_face',i]);valid.append(True)
 data=np.asarray(rows,np.float32);expected=np.asarray(expected,np.int32);oracle=[]
 for row,e,ok in zip(data,expected,valid):
  if not ok:oracle.append(None);continue
  signs=[orient(*row[:3],q) for q in itertools.product(*zip(row[4],row[5]))]
  all_inside=all(s<=0 for s in signs);oracle.append(all_inside)
  if e:assert all_inside
  if not e and np.all(row[0]==row[1]):raise AssertionError('unexpected degenerate face')
 w=Path('.work/vtm');npz=w/'axis-hull-primitive-v1.npz';dest=w/'axis-hull-primitive-v1.json';assert not npz.exists() and not dest.exists()
 np.savez_compressed(npz,data=data,expected=expected,valid_preconditions=np.asarray(valid))
 wp.init();wp.set_device('cuda:0');gpu=wp.array(data,dtype=wp.vec3);out=wp.zeros(len(data),dtype=int)
 with wp.ScopedCapture() as capture:wp.launch(evaluate,len(data),inputs=[gpu,out])
 wp.capture_launch(capture.graph);actual=out.numpy();mismatch=np.flatnonzero(actual!=expected).tolist()
 report=dict(scope=__doc__,cases=len(data),accepted=int(actual.sum()),mismatches=mismatch,names=names,oracle_box_inside=oracle,input_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),w/'axis_hull_support_v1.py']})
 dest.write_text(json.dumps(report,indent=2)+'\n');print('cases',len(data),'accepted',actual.sum(),'mismatches',mismatch,flush=True)
 assert not mismatch,[(i,names[i],data[i].tolist(),int(actual[i]),int(expected[i])) for i in mismatch]

if __name__=='__main__':main()
