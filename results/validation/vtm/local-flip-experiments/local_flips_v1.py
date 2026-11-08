"""Private bounded bistellar candidate. Every published result has a full certificate.

Only strictly convex five-site cavities are changed. A parallel round reserves
all old cells before emitting into separate buffers. This experiment deliberately
recertifies after every round; it is not a standalone API or adopted runtime.
"""
import warp as wp
from warpower import _filtered_predicates as fp
from warpower._delaunay_certificate import DelaunayCertificate,symbolic_sphere
from warpower._domain import _Scan
from warpower._mesh_topology import face_vertices,sorted_face
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False,'block_dim':32})

@wp.func
def positive(points:wp.array[wp.vec3],a:int,b:int,c:int,d:int):
 ids=wp.vec4i(a,b,c,d);s=fp.orient3d(points[a],points[b],points[c],points[d])
 if s.valid==0 or s.sign==0:return wp.vec4i(-1)
 if s.sign<0:ids[0]=b;ids[1]=a
 return ids

@wp.func
def convex(points:wp.array[wp.vec3],a:int,b:int,c:int,d:int,e:int):
 # The tips lie on opposite sides; their open segment pierces the open triangle.
 s=fp.orient3d(points[a],points[b],points[c],points[d])
 t=fp.orient3d(points[a],points[b],points[c],points[e])
 if s.valid==0 or t.valid==0 or s.sign*t.sign!=-1:return False
 u=fp.orient3d(points[d],points[e],points[a],points[b])
 v=fp.orient3d(points[d],points[e],points[b],points[c])
 w=fp.orient3d(points[d],points[e],points[c],points[a])
 return u.valid!=0 and v.valid!=0 and w.valid!=0 and u.sign!=0 and u.sign==v.sign and u.sign==w.sign

@wp.func
def strict(points:wp.array[wp.vec3],ids:wp.vec4i,query:int):
 if ids[0]<0:return False
 s=symbolic_sphere(points,ids,query)
 return s.valid!=0 and s.sign<0

@wp.func
def opposite(tets:wp.array2d[int],cell:int,vertex:int):
 result=int(-1)
 for k in range(4):
  if tets[cell,k]==vertex:result=4*cell+k
 return result

@wp.kernel
def twins_kernel(tets:wp.array2d[int],count:wp.array[int],table:wp.array[int],twins:wp.array[int],error:wp.array[int]):
 face=wp.tid()
 if face>=4*count[0]:return
 key=sorted_face(face_vertices(tets,face))
 slot=int((wp.uint32(key[0])*wp.uint32(73856093)^wp.uint32(key[1])*wp.uint32(19349663)^wp.uint32(key[2])*wp.uint32(83492791))&wp.uint32(table.shape[0]-1))
 for _ in range(table.shape[0]):
  other=table[slot]
  if other<0:wp.atomic_or(error,0,1024);return
  if sorted_face(face_vertices(tets,other))==key:
   if other!=face:twins[face]=other;twins[other]=face
   return
  slot=(slot+1)&(table.shape[0]-1)
 wp.atomic_or(error,0,1024)

@wp.kernel
def propose(points:wp.array[wp.vec3],tets:wp.array2d[int],count:wp.array[int],twins:wp.array[int],mode:wp.array[int],old:wp.array[wp.vec3i],new:wp.array2d[wp.vec4i],owner:wp.array[int],statistics:wp.array[int],error:wp.array[int]):
 face=wp.tid();mode[face]=0
 if face>=4*count[0] or error[0]!=0:return
 paired=twins[face]
 if paired<0 or face<paired:return
 i=face//4;j=paired//4
 ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3]);e=tets[j,paired%4]
 s=symbolic_sphere(points,ids,e)
 if s.valid==0:wp.atomic_or(error,0,2);return
 if s.sign<=0:return
 f=face_vertices(tets,face);a=f[0];b=f[1];c=f[2];d=tets[i,face%4]
 kind=int(0);cells=wp.vec3i(i,j,-1);n0=wp.vec4i(-1);n1=wp.vec4i(-1);n2=wp.vec4i(-1)
 if convex(points,a,b,c,d,e):
  n0=positive(points,a,b,d,e);n1=positive(points,b,c,d,e);n2=positive(points,c,a,d,e)
  if strict(points,n0,c) and strict(points,n1,a) and strict(points,n2,b):kind=3
 if kind==0:
  # The third cell must be the common neighbor across the two remaining
  # edge faces. This proves a closed three-cell link of the candidate edge.
  for k in range(3):
   u=f[k];v=f[(k+1)%3];r=f[(k+2)%3]
   fi=opposite(tets,i,r);fj=opposite(tets,j,r)
   if fi<0 or fj<0:continue
   ni=twins[fi];nj=twins[fj]
   if ni<0 or nj<0 or ni//4!=nj//4:continue
   third=ni//4
   if third==i or third==j:continue
   expected=wp.vec4i(u,v,d,e);membership=bool(True)
   for q in range(4):
    found=bool(False)
    for z in range(4):found=found or tets[third,z]==expected[q]
    membership=membership and found
   if not membership or not convex(points,r,d,e,u,v):continue
   n0=positive(points,r,d,e,u);n1=positive(points,r,d,e,v)
   if strict(points,n0,v) and strict(points,n1,u):
    kind=2;cells=wp.vec3i(i,j,third);break
 if kind==0:return
 mode[face]=kind;old[face]=cells;new[face,0]=n0;new[face,1]=n1;new[face,2]=n2
 wp.atomic_add(statistics,3-kind,1)
 wp.atomic_min(owner,cells[0],face);wp.atomic_min(owner,cells[1],face)
 if kind==2:wp.atomic_min(owner,cells[2],face)

@wp.kernel
def choose(mode:wp.array[int],old:wp.array[wp.vec3i],owner:wp.array[int],keep:wp.array[int],emit:wp.array[int],statistics:wp.array[int]):
 f=wp.tid();emit[f]=0;kind=mode[f]
 if kind==0:return
 cells=old[f]
 for k in range(5-kind):
  if owner[cells[k]]!=f:return
 for k in range(5-kind):keep[cells[k]]=0
 emit[f]=kind;wp.atomic_add(statistics,5-kind,1)

@wp.kernel
def init_keep(count:wp.array[int],keep:wp.array[int]):
 i=wp.tid();keep[i]=int(i<count[0])

@wp.kernel
def size_output(prefix:wp.array[int],new_prefix:wp.array[int],count:wp.array[int],ready:wp.array[int],error:wp.array[int],rounds:wp.array[int],stats:wp.array[int],history:wp.array2d[int]):
 m=prefix.shape[0];kept=prefix[m-1];added=new_prefix[new_prefix.shape[0]-1];total=kept+added
 at=rounds[0];rounds[0]+=1
 for k in range(4):history[at,k]=stats[k]
 history[at,4]=kept;history[at,5]=added;history[at,6]=total
 if added==0:error[0]|=512
 if total>m:error[0]|=256
 ready[0]=int(error[0]==0);count[0]=0
 if ready[0]!=0:count[0]=total

@wp.kernel
def emit_kept(tets:wp.array2d[int],keep:wp.array[int],prefix:wp.array[int],output:wp.array2d[int]):
 i=wp.tid()
 if keep[i]!=0:
  for k in range(4):output[prefix[i]-1,k]=tets[i,k]

@wp.kernel
def emit_new(cells:wp.array2d[wp.vec4i],emit:wp.array[int],prefix:wp.array[int],kept:wp.array[int],output:wp.array2d[int]):
 f=wp.tid();n=emit[f]
 start=kept[kept.shape[0]-1]+prefix[f]-n
 for j in range(n):
  v=cells[f,j]
  for k in range(4):output[start+j,k]=v[k]

@wp.kernel
def transition(cert_error:wp.array[int],input_count:wp.array[int],count:wp.array[int],rounds:wp.array[int],max_rounds:int,error:wp.array[int],accepted:wp.array[int],running:wp.array[int]):
 error[0]=cert_error[0];accepted[0]=0;running[0]=0
 if input_count[0]!=count[0]:error[0]|=2048
 if error[0]==0:accepted[0]=1
 elif error[0]==8:
  if rounds[0]<max_rounds:running[0]=1
  else:error[0]|=4096

@wp.kernel
def finish(error:wp.array[int],accepted:wp.array[int],count:wp.array[int]):
 if accepted[0]==0:
  count[0]=0
  if error[0]==0:error[0]=8192

class Candidate:
 def __init__(self,points,input_tets,input_count,tree,max_rounds=16):
  self.points=points;self.input_tets=input_tets;self.input_count=input_count;self.tree=tree;self.device=points.device;self.max_rounds=max_rounds
  m=len(input_tets)
  with wp.ScopedDevice(self.device):
   self.work=wp.empty_like(input_tets);self.work_count=wp.zeros(1,dtype=int)
   self.cert=DelaunayCertificate(points,self.work,self.work_count,tree)
   self.tetrahedra=self.cert.tetrahedra;self.tetrahedra_count=wp.zeros(1,dtype=int)
   self.error=wp.zeros(1,dtype=int);self.accepted=wp.zeros(1,dtype=int);self.running=wp.zeros(1,dtype=int);self.ready=wp.zeros(1,dtype=int);self.rounds=wp.zeros(1,dtype=int)
   self.twins=wp.empty(4*m,dtype=int);self.mode=wp.empty(4*m,dtype=int);self.old=wp.empty(4*m,dtype=wp.vec3i);self.new=wp.empty((4*m,3),dtype=wp.vec4i);self.owner=wp.empty(m,dtype=int)
   self.keep=wp.empty(m,dtype=int);self.prefix=wp.empty(m,dtype=int);self.emit=wp.empty(4*m,dtype=int);self.new_prefix=wp.empty(4*m,dtype=int);self.scan=_Scan(m,self.device);self.scan_new=_Scan(4*m,self.device)
   self.stats=wp.zeros(4,dtype=int);self.history=wp.zeros((max_rounds,7),dtype=int)
   wp.load_module(module=__name__,device=self.device)
 def launch(self,k,dim,args):wp.launch(k,dim,inputs=args,device=self.device,block_dim=32)
 def certify(self):
  self.cert.run()
  self.launch(transition,1,[self.cert.error,self.work_count,self.cert.tetrahedra_count,self.rounds,self.max_rounds,self.error,self.accepted,self.running])
 def run(self):
  wp.copy(self.work,self.input_tets);wp.copy(self.work_count,self.input_count);self.rounds.zero_();self.history.zero_();self.stats.zero_();self.accepted.zero_();self.tetrahedra_count.zero_()
  self.certify()
  wp.capture_while(self.running,self.step)
  wp.copy(self.tetrahedra_count,self.cert.tetrahedra_count)
  self.launch(finish,1,[self.error,self.accepted,self.tetrahedra_count])
 def step(self):
  m=len(self.work);self.running.zero_();self.error.zero_();self.stats.zero_();self.twins.fill_(-1);self.owner.fill_(2147483647)
  self.launch(twins_kernel,4*m,[self.cert.tetrahedra,self.cert.tetrahedra_count,self.cert._table,self.twins,self.error])
  self.launch(propose,4*m,[self.points,self.cert.tetrahedra,self.cert.tetrahedra_count,self.twins,self.mode,self.old,self.new,self.owner,self.stats,self.error])
  self.launch(init_keep,m,[self.cert.tetrahedra_count,self.keep])
  self.launch(choose,4*m,[self.mode,self.old,self.owner,self.keep,self.emit,self.stats])
  self.scan.run(self.keep,self.prefix);self.scan_new.run(self.emit,self.new_prefix)
  self.launch(size_output,1,[self.prefix,self.new_prefix,self.work_count,self.ready,self.error,self.rounds,self.stats,self.history])
  wp.capture_if(self.ready,self.write_and_check)
 def write_and_check(self):
  self.launch(emit_kept,len(self.work),[self.cert.tetrahedra,self.keep,self.prefix,self.work])
  self.launch(emit_new,4*len(self.work),[self.new,self.emit,self.new_prefix,self.prefix,self.work])
  self.certify()
