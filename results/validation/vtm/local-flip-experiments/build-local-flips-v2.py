from pathlib import Path
w=Path('.work/vtm');s=(w/'local_flips_v1.py').read_text()
s=s.replace('This experiment deliberately\nrecertifies after every round; it is not a standalone API or adopted runtime.','This revision checks positive geometry, coverage and indexed topology per round,\nand certifies the full result once before publication. Initial hull support is\nchecked before any flips. It is not a public API or adopted runtime.')
s=s.replace('DelaunayCertificate,symbolic_sphere','DelaunayCertificate,symbolic_sphere,select_positive,compact,ready,coverage,collect_boundary_faces')
s=s.replace('face_vertices,sorted_face','face_vertices,sorted_face,identity,index_faces')
s=s.replace('statistics:wp.array[int],error:wp.array[int]):\n face=wp.tid();mode[face]=0','statistics:wp.array[int],error:wp.array[int],violations:wp.array[int]):\n face=wp.tid();mode[face]=0')
s=s.replace('if s.sign<=0:return\n f=face_vertices','if s.sign<=0:return\n wp.atomic_add(violations,0,1)\n f=face_vertices')
a=s.index('@wp.kernel\ndef size_output');b=s.index('@wp.kernel\ndef emit_kept',a)
s=s[:a]+'''@wp.kernel
def size_output(prefix:wp.array[int],new_prefix:wp.array[int],count:wp.array[int],ready:wp.array[int],error:wp.array[int],rounds:wp.array[int],stats:wp.array[int],history:wp.array2d[int],violations:wp.array[int],done:wp.array[int],max_rounds:int):
 m=prefix.shape[0];kept=prefix[m-1];added=new_prefix[new_prefix.shape[0]-1];total=kept+added
 ready[0]=0;done[0]=0
 if error[0]!=0:return
 if violations[0]==0:done[0]=1;return
 if added==0:error[0]=512;return
 at=rounds[0]
 if at>=max_rounds:error[0]=4096;return
 for k in range(4):history[at,k]=stats[k]
 history[at,4]=kept;history[at,5]=added;history[at,6]=total
 if total>m:error[0]=256;return
 rounds[0]+=1;ready[0]=1;count[0]=total

@wp.kernel
def geometry_ready(error:wp.array[int],expected:wp.array[int],count:wp.array[int],out_error:wp.array[int],running:wp.array[int]):
 out_error[0]=error[0]
 if expected[0]!=count[0]:out_error[0]|=2048
 running[0]=int(out_error[0]==0)

@wp.kernel
def certified(error:wp.array[int],expected:wp.array[int],count:wp.array[int],out_error:wp.array[int],accepted:wp.array[int]):
 out_error[0]=error[0]
 if expected[0]!=count[0]:out_error[0]|=2048
 accepted[0]=int(out_error[0]==0)

'''+s[b:]
a=s.index(' def certify(self):')
s=s[:a]+''' def geometry(self):
  c=self.cert;tree=self.tree;c.error.zero_();c.statistics.zero_();c._used.zero_()
  c._launch(select_positive,len(c.input_tets),[c.points,c.input_tets,c.input_count,tree.owners,c._selected,c.error])
  c._scan.run(c._selected,c._prefix)
  c._launch(compact,len(c.input_tets),[c.input_tets,c._selected,c._prefix,tree.owners,c.tetrahedra,c.tetrahedra_count,c._used])
  c._launch(ready,1,[c.error,c._ready]);wp.capture_if(c._ready,self.index)
  c._launch(coverage,len(c.points),[tree.owners,c._used,c.tetrahedra_count,c.error])
  self.launch(geometry_ready,1,[c.error,self.work_count,c.tetrahedra_count,self.error,self.running])
 def index(self):
  c=self.cert;c._table.fill_(-1);c._occurrences.zero_();c._balance.zero_();c._boundary_count.zero_()
  c._launch(identity,len(c._parents),[c._parents])
  c._launch(index_faces,4*len(c.tetrahedra),[c.tetrahedra,c.tetrahedra_count,c._table,c._occurrences,c._balance,c._parents])
  c._launch(collect_boundary_faces,len(c._table),[c._table,c._occurrences,c._balance,c._boundary,c._boundary_count,c.error])
 def hull(self):
  self.cert._hull_support(self.tree)
  self.launch(geometry_ready,1,[self.cert.error,self.work_count,self.cert.tetrahedra_count,self.error,self.running])
 def certify(self):
  self.cert.run()
  self.launch(certified,1,[self.cert.error,self.work_count,self.cert.tetrahedra_count,self.error,self.accepted])
 def run(self):
  wp.copy(self.work,self.input_tets);wp.copy(self.work_count,self.input_count);self.rounds.zero_();self.history.zero_();self.stats.zero_();self.accepted.zero_();self.done.zero_();self.tetrahedra_count.zero_()
  self.geometry();wp.capture_if(self.running,self.hull)
  wp.capture_while(self.running,self.step)
  wp.capture_if(self.done,self.certify)
  wp.copy(self.tetrahedra_count,self.cert.tetrahedra_count)
  self.launch(finish,1,[self.error,self.accepted,self.tetrahedra_count])
 def step(self):
  m=len(self.work);self.running.zero_();self.error.zero_();self.stats.zero_();self.violations.zero_();self.twins.fill_(-1);self.owner.fill_(2147483647)
  self.launch(twins_kernel,4*m,[self.cert.tetrahedra,self.cert.tetrahedra_count,self.cert._table,self.twins,self.error])
  self.launch(propose,4*m,[self.points,self.cert.tetrahedra,self.cert.tetrahedra_count,self.twins,self.mode,self.old,self.new,self.owner,self.stats,self.error,self.violations])
  self.launch(init_keep,m,[self.cert.tetrahedra_count,self.keep])
  self.launch(choose,4*m,[self.mode,self.old,self.owner,self.keep,self.emit,self.stats])
  self.scan.run(self.keep,self.prefix);self.scan_new.run(self.emit,self.new_prefix)
  self.launch(size_output,1,[self.prefix,self.new_prefix,self.work_count,self.ready,self.error,self.rounds,self.stats,self.history,self.violations,self.done,self.max_rounds])
  wp.capture_if(self.ready,self.write_and_check)
 def write_and_check(self):
  self.launch(emit_kept,len(self.work),[self.cert.tetrahedra,self.keep,self.prefix,self.work])
  self.launch(emit_new,4*len(self.work),[self.new,self.emit,self.new_prefix,self.prefix,self.work])
  self.geometry()
'''
s=s.replace('self.stats=wp.zeros(4,dtype=int);','self.violations=wp.zeros(1,dtype=int);self.done=wp.zeros(1,dtype=int);self.stats=wp.zeros(4,dtype=int);')
p=w/'local_flips_v2.py';assert not p.exists();p.write_text(s)
s=(w/'run-local-flips-v1.py').read_text().replace('local_flips_v1','local_flips_v2');(w/'run-local-flips-v2.py').write_text(s)
s=(w/'local_flip_mesh_v1.py').read_text().replace('local_flips_v1','local_flips_v2');(w/'local_flip_mesh_v2.py').write_text(s)
s=(w/'run-local-flip-gallery-v1.py').read_text().replace('local_flip_mesh_v1','local_flip_mesh_v2').replace('local_flips_v1','local_flips_v2');(w/'run-local-flip-gallery-v2.py').write_text(s)
