"""Original/grouped accumulation: ownership, error paths, maxima and bounded sum differences."""
import hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from warpower._site_bvh import SiteHierarchy
from warpower._boundary import accumulate_samples
from boundary_global_reduce_kernels_v1 import accumulate_reduced
out=Path('.work/vtm/boundary-global-reduce-check-v1.json');assert not out.exists();rng=np.random.default_rng(223);rows=[]

def check(label,points,hierarchy,samples,weights,sizing,dimension,active_count=None,initial_error=0):
 n=len(samples);component=dimension-1;active=wp.full(1,len(points) if active_count is None else active_count,dtype=int);occupied=wp.full(len(points),-1,dtype=int)
 sample=wp.array(samples,dtype=wp.vec3);base=wp.array(weights,dtype=wp.float64);sizes=wp.array(sizing,dtype=float);answers=[]
 for kernel in [accumulate_samples,accumulate_reduced]:
  owners=wp.full(n,77,dtype=int);mass=wp.zeros(len(points),dtype=wp.float64);moment=wp.zeros(len(points),dtype=wp.vec3d)
  values=np.full(2,77.);values[component]=0
  energy=wp.array(values,dtype=wp.float64);total=wp.array(values,dtype=wp.float64);maximum=wp.array(values,dtype=wp.float64);error=wp.full(1,initial_error,dtype=int)
  wp.launch(kernel,n,inputs=[hierarchy.tree,points,active,occupied,sample,base,sizes,dimension,owners,mass,moment,energy,total,maximum,component,error],block_dim=32)
  answers.append(dict(owners=owners.numpy(),mass=mass.numpy(),moment=moment.numpy(),energy=energy.numpy(),total=total.numpy(),maximum=maximum.numpy(),error=error.numpy()))
 a,b=answers;np.testing.assert_array_equal(a['error'],b['error'])
 for d in answers:
  for k in ['energy','total','maximum']:assert d[k][1-component]==77
 record=dict(label=label,n=n,dimension=dimension,error=int(a['error'][0]),error_equal=True,other_component_untouched=True)
 if a['error'][0]==0:
  np.testing.assert_array_equal(a['owners'],b['owners']);np.testing.assert_array_equal(a['maximum'].view(np.uint64),b['maximum'].view(np.uint64))
  p=points.numpy().astype(np.float64);valid=a['owners']>=0;owner=a['owners'][valid]
  weight=weights[valid]/sizing[valid].astype(np.float64)**(4 if dimension==2 else 3)
  delta=samples[valid].astype(np.float64)-p[owner];energy_terms=weight*np.sum(delta*delta,axis=1)
  eps=np.finfo(np.float64).eps;factor=16*(n+4)*eps
  absolute={'mass':np.zeros(len(points)), 'moment':np.zeros((len(points),3)), 'energy':float(np.sum(np.abs(energy_terms))), 'total':float(np.sum(np.abs(weight)))}
  np.add.at(absolute['mass'],owner,np.abs(weight));np.add.at(absolute['moment'],owner,np.abs(weight[:,None]*delta))
  differences={}
  for k in ['mass','moment','energy','total']:
   x,y=(a[k][component],b[k][component]) if k in ['energy','total'] else (a[k],b[k])
   difference=np.abs(x-y);bound=factor*absolute[k]+np.finfo(np.float64).tiny
   assert np.all(difference<=bound),(label,n,dimension,k,difference,bound)
   differences[k]=dict(maximum_absolute_difference=float(np.max(difference)),maximum_absolute_envelope=float(np.max(bound)))
  record.update(owners_equal=True,maximum_distance_bits_equal=True,sum_comparison=differences)
 rows.append(record);out.write_text(json.dumps(dict(scope=__doc__,cases=rows,complete=False),indent=2)+'\n');print(label,n,dimension,record['error'],flush=True)

for scale in [1e-30,1.,1e30]:
 points=wp.array((rng.normal(size=(41,3))*scale).astype(np.float32),dtype=wp.vec3);hierarchy=SiteHierarchy(points)
 for n in [1,5,31,32,33,97,1057]:
  samples=(rng.normal(size=(n,3))*scale).astype(np.float32);weights=rng.uniform(.1,2,n)*scale*scale;sizing=rng.uniform(.3,1.3,n).astype(np.float32)
  for kind in ['dense','zeros']:
   ww=weights.copy()
   if kind=='zeros':ww[:min(32,n)]=0;ww[::3]=0
   for dimension in [1,2]:check(str(scale)+'-'+kind,points,hierarchy,samples,ww,sizing,dimension)
points=wp.array(rng.normal(size=(41,3)).astype(np.float32),dtype=wp.vec3);hierarchy=SiteHierarchy(points)
for bad in ['sizing-zero','sizing-nan','weight-inf','sample-nan','empty-active','initial-error']:
 samples=rng.normal(size=(33,3)).astype(np.float32);weights=np.ones(33);sizing=np.ones(33,np.float32)
 if bad=='sizing-zero':sizing[0]=0
 if bad=='sizing-nan':sizing[0]=np.nan
 if bad=='weight-inf':weights[0]=np.inf
 if bad=='sample-nan':samples[0]=np.nan
 check(bad,points,hierarchy,samples,weights,sizing,2,active_count=0 if bad=='empty-active' else None,initial_error=4 if bad=='initial-error' else 0)
r=dict(scope=__doc__,cases=rows,complete=True,comparison_note='Conservative16*(N+4)*eps*sumabs envelope for finite test contributions, including arithmetic/context and summation differences; not a general proof or bitwise sum identity.',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/boundary_global_reduce_kernels_v1.py'),Path('warpower/_boundary.py')]});out.write_text(json.dumps(r,indent=2)+'\n');print(len(rows),'cases passed',flush=True)
