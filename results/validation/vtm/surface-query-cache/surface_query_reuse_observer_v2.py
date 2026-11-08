"""Observe exact repeated queries; every original surface query still runs."""
import json
from pathlib import Path
import warp as wp
from warpower import _domain, _extraction, _selective, _quality_moves, _mesher
from surface_query_reuse_kernels_v1 import QueryState, observe_full, observe_retained

wp.set_module_options({'enable_backward': False, 'fast_math': False})
instances=[]
COLUMNS=['calls','geometry_queries','same_row_centers','keyed_centers','ray_queries','same_row_rays','keyed_rays','same_row_ray_mismatches','keyed_ray_mismatches','nearest_queries','same_row_nearest_centers','keyed_nearest_centers','same_row_identical_nearest','keyed_identical_nearest','same_row_nearest_mismatches','keyed_nearest_mismatches','table_insert_overflow','table_lookup_overflow']

@wp.func
def key(bits:wp.array[wp.int64],i:int,mask:int):
    h=wp.uint64(bits[3*i])*wp.uint64(0x9E3779B97F4A7C15)
    h=h^(wp.uint64(bits[3*i+1])*wp.uint64(0xBF58476D1CE4E5B9))
    h=h^(wp.uint64(bits[3*i+2])*wp.uint64(0x94D049BB133111EB))
    h=(h^(h>>wp.uint64(30)))*wp.uint64(0xBF58476D1CE4E5B9)
    h=(h^(h>>wp.uint64(27)))*wp.uint64(0x94D049BB133111EB)
    h=h^(h>>wp.uint64(31))
    return int(h&wp.uint64(mask))

@wp.func
def same(a:wp.array[wp.int64],i:int,b:wp.array[wp.int64],j:int):
    return a[3*i]==b[3*j] and a[3*i+1]==b[3*j+1] and a[3*i+2]==b[3*j+2]

@wp.kernel
def build(valid:wp.array[int],bits:wp.array[wp.int64],table:wp.array[int],stats:wp.array[wp.int64]):
    i=wp.tid()
    if valid[i]==0:return
    mask=table.shape[0]-1;slot=key(bits,i,mask)
    for _ in range(128):
        old=wp.atomic_cas(table,slot,-1,i)
        if old<0:return
        if same(bits,i,bits,old):
            wp.atomic_min(table,slot,i)
            return
        slot=(slot+1)&mask
    wp.atomic_add(stats,16,wp.int64(1))

@wp.kernel
def measure(current:QueryState,previous:QueryState,cb:wp.array[wp.int64],pb:wp.array[wp.int64],cl:wp.array[wp.int64],pl:wp.array[wp.int64],cd:wp.array[wp.int64],pd:wp.array[wp.int64],cp:wp.array[wp.int64],pp:wp.array[wp.int64],table:wp.array[int],stats:wp.array[wp.int64]):
    i=wp.tid()
    if i==0:wp.atomic_add(stats,0,wp.int64(1))
    if current.valid[i]==0:return
    wp.atomic_add(stats,1,wp.int64(1))
    if current.rays[i]!=0:wp.atomic_add(stats,4,wp.int64(1))
    if current.nearest[i]!=0:wp.atomic_add(stats,9,wp.int64(1))
    direct=int(-1)
    if previous.valid[i]!=0 and same(cb,i,pb,i):direct=i
    found=int(-1);slot=key(cb,i,table.shape[0]-1);stopped=bool(False)
    for _ in range(128):
        j=table[slot]
        if j<0:
            stopped=True
            break
        if same(cb,i,pb,j):
            found=j;stopped=True
            break
        slot=(slot+1)&(table.shape[0]-1)
    if not stopped:wp.atomic_add(stats,17,wp.int64(1))
    for kind in range(2):
        j=wp.where(kind==0,direct,found)
        if j<0:continue
        wp.atomic_add(stats,2+kind,wp.int64(1))
        if current.rays[i]!=0 and previous.rays[j]!=0:
            wp.atomic_add(stats,5+kind,wp.int64(1))
            if current.sides[i]!=previous.sides[j]:wp.atomic_add(stats,7+kind,wp.int64(1))
        if current.nearest[i]!=0 and previous.nearest[j]!=0:
            wp.atomic_add(stats,10+kind,wp.int64(1))
            if cl[i]==pl[j]:
                wp.atomic_add(stats,12+kind,wp.int64(1))
                if current.faces[i]!=previous.faces[j] or cd[i]!=pd[j] or not same(cp,i,pp,j):
                    wp.atomic_add(stats,14+kind,wp.int64(1))

FIELDS={'centers':wp.vec3d,'valid':int,'rays':int,'sides':int,'nearest':int,'limits':wp.float64,'distances':wp.float64,'faces':int,'hit_points':wp.vec3d}

class Observation:
    def __init__(self,domain,capacity,label,site_capacity):
        self.domain=domain;self.capacity=capacity;self.label=label;self.site_capacity=site_capacity
        self.device=domain.device
        with wp.ScopedDevice(self.device):
            self.current=QueryState();self.previous=QueryState()
            for state in [self.current,self.previous]:
                for name,dtype in FIELDS.items():setattr(state,name,wp.zeros(capacity,dtype=dtype))
            self.table=wp.full(1<<(4*capacity-1).bit_length(),-1,dtype=int)
            self.stats=wp.zeros(len(COLUMNS),dtype=wp.int64)
            wp.load_module(module='surface_query_reuse_kernels_v1',device=self.device,block_dim=32)
            wp.load_module(module=__name__,device=self.device)
        instances.append(self)

    def reset(self):
        self.previous.valid.zero_();self.table.fill_(-1)

    def after(self):
        c,p=self.current,self.previous
        args=[c,p]
        for field in ['centers','limits','distances','hit_points']:
            args.extend([getattr(state,field).view(wp.int64).flatten() for state in [c,p]])
        wp.launch(measure,self.capacity,inputs=[*args,self.table,self.stats],device=self.device)
        for field in FIELDS:wp.copy(getattr(p,field),getattr(c,field))
        self.table.fill_(-1)
        wp.launch(build,self.capacity,inputs=[p.valid,p.centers.view(wp.int64).flatten(),self.table,self.stats],device=self.device)

def install():
    cls=_extraction.ExtractionWorkspace;original_init=cls.__init__;original_launch=cls._launch
    def init(self,domain,points,tetrahedra,*args,**kwargs):
        self._surface_observation=Observation(domain,len(tetrahedra),'full-extraction-'+type(self).__name__,len(points))
        original_init(self,domain,points,tetrahedra,*args,**kwargs)
    def launch(self,kernel,dim,args,*,block_dim=256):
        if kernel is _extraction.classify:
            o=self._surface_observation
            wp.launch(observe_full,dim,inputs=[*args,o.current],device=self.device,block_dim=block_dim)
            o.after()
        else:original_launch(self,kernel,dim,args,block_dim=block_dim)
    cls.__init__=init;cls._launch=launch
    for cls,label in [(_selective.SelectiveUpdate,'retained-selection'),(_quality_moves.QualityMove,'retained-quality')]:
        def patch(cls,label):
            old_init,old_launch=cls.__init__,cls._launch
            def init(self,core,*args,**kwargs):
                self._surface_observation=Observation(core.domain,len(core.full_tetrahedra),label+'-'+type(self).__name__,core.n)
                old_init(self,core,*args,**kwargs)
            def launch(self,kernel,dim,args,*,block_dim=256):
                if kernel is _selective.classify_retained:
                    o=self._surface_observation
                    wp.launch(observe_retained,dim,inputs=[*args,o.current],device=self.device,block_dim=block_dim)
                    o.after()
                else:old_launch(self,kernel,dim,args,block_dim=block_dim)
            cls.__init__=init;cls._launch=launch
        patch(cls,label)
    old_refit=_domain.SurfaceDomain.refit
    def refit(self,*args,**kwargs):
        for o in instances:
            if o.domain is self:o.reset()
        return old_refit(self,*args,**kwargs)
    _domain.SurfaceDomain.refit=refit
    old_record=_mesher.VariationalMeshingWorkspace._record
    def record(self):
        for o in instances:
            if o.domain is self.domain:o.reset()
        old_record(self)
    _mesher.VariationalMeshingWorkspace._record=record
    return instances

def report(path):
    rows=[]
    for o in instances:
        stats=o.stats.numpy().tolist()
        rows.append(dict(label=o.label,site_capacity=o.site_capacity,tet_capacity=o.capacity,statistics=stats))
    p=Path(path);assert not p.exists()
    p.write_text(json.dumps(dict(scope=__doc__+' Counts cover all replays and are diagnostic, not speed evidence. Previous-call caches reset at fresh full solve/domain refit; every original query still executes. Keyed duplicates choose the minimum previous row. Bounded nearest reuse requires identical cutoff bits.',columns=COLUMNS,instances=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)
