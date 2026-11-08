"""Private prototype: previous-query results under an unchanged surface epoch."""
import warp as wp
from warpower import _domain,_extraction,_selective,_quality_moves,_mesher
from surface_query_cache_kernels_v1 import QueryState,build_cache,classify_cached,retained_cached,selection_cached

FIELDS={'centers':wp.vec3d,'valid':int,'rays':int,'sides':int,'nearest':int,'limits':wp.float64,'distances':wp.float64,'faces':int,'hit_points':wp.vec3d}
instances=[]

class Cache:
    def __init__(self,domain,capacity,label):
        self.domain=domain;self.device=domain.device;self.capacity=capacity;self.label=label
        with wp.ScopedDevice(self.device):
            self.current=QueryState();self.previous=QueryState()
            for state in [self.current,self.previous]:
                for field,dtype in FIELDS.items():setattr(state,field,wp.zeros(capacity,dtype=dtype))
            self.table=wp.full(1<<(4*capacity-1).bit_length(),-1,dtype=int)
            wp.load_module(module='surface_query_cache_kernels_v1',device=self.device,block_dim=32)
        instances.append(self)

    def reset(self):self.table.fill_(-1)

    def after(self,error):
        for field in FIELDS:wp.copy(getattr(self.previous,field),getattr(self.current,field))
        self.table.fill_(-1)
        wp.launch(build_cache,self.capacity,inputs=[self.previous,self.table,error],device=self.device)

    def launch(self,kernel,dim,args,block_dim=32):
        wp.launch(kernel,dim,inputs=[*args,self.current,self.previous,self.table],device=self.device,block_dim=block_dim)
        self.after(args[5])

def install():
    cls=_extraction.ExtractionWorkspace;old_init,old_launch=cls.__init__,cls._launch
    def init(self,domain,points,tetrahedra,*args,**kwargs):
        self._query_cache=Cache(domain,len(tetrahedra),type(self).__name__)
        old_init(self,domain,points,tetrahedra,*args,**kwargs)
    def launch(self,kernel,dim,args,*,block_dim=256):
        if kernel in [_extraction.classify,_extraction.classify_selection]:
            target=classify_cached if kernel is _extraction.classify else selection_cached
            self._query_cache.launch(target,dim,args,block_dim)
        else:old_launch(self,kernel,dim,args,block_dim=block_dim)
    cls.__init__=init;cls._launch=launch
    for cls in [_selective.SelectiveUpdate,_quality_moves.QualityMove]:
        def patch(cls):
            original_init,original_launch=cls.__init__,cls._launch
            def init(self,core,*args,**kwargs):
                self._query_cache=Cache(core.domain,len(core.full_tetrahedra),type(self).__name__)
                original_init(self,core,*args,**kwargs)
            def launch(self,kernel,dim,args,*,block_dim=256):
                if kernel is _selective.classify_retained:self._query_cache.launch(retained_cached,dim,args,block_dim)
                else:original_launch(self,kernel,dim,args,block_dim=block_dim)
            cls.__init__=init;cls._launch=launch
        patch(cls)
    original_refit=_domain.SurfaceDomain.refit
    def refit(self,*args,**kwargs):
        for c in instances:
            if c.domain is self:c.reset()
        return original_refit(self,*args,**kwargs)
    _domain.SurfaceDomain.refit=refit
    original_record=_mesher.VariationalMeshingWorkspace._record
    def record(self):
        for c in instances:
            if c.domain is self.domain:c.reset()
        original_record(self)
    _mesher.VariationalMeshingWorkspace._record=record
    return instances
