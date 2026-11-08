"""Prepare an unexecuted scalar-query ordering probe without altering geometric expressions."""
import ast,copy,hashlib,json
from pathlib import Path
w=Path('.work/vtm');source=Path('warpower/_surface_query_cache.py');tree=ast.parse(source.read_text());names=['classify_cached','selection_cached','retained_cached']
header='''"""Private issue197 centroid-ordered scalar queries. Not yet executed or qualified."""
import warp as wp
from warp.utils import radix_sort_pairs
from warpower import _surface_query_cache as original
from warpower._surface_query_cache import *
from warpower._bvh import expand_bits

wp.set_module_options({"enable_backward": False, "fast_math": False})

@wp.kernel
def centroid_keys(tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int], count: wp.array[int], error: wp.array[int], keys: wp.array[wp.uint64], ids: wp.array[int]):
    i=wp.tid()
    ids[i]=i
    keys[i]=wp.uint64(18446744073709551615)
    if error[0]!=0 or tree.n<=0 or i>=count[0] or i>=tets.shape[0]:
        return
    center=wp.vec3d(0.0)
    for k in range(4):
        site=tets[i,k]
        if site<0 or site>=points.shape[0]:
            return
        p=points[site]
        if not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):
            return
        center+=wp.vec3d(p)*wp.float64(0.25)
    lower=wp.vec3d(tree.lower[1]);upper=wp.vec3d(tree.upper[1]);q=wp.vec3d(0.0)
    for k in range(3):
        if not wp.isfinite(lower[k]) or not wp.isfinite(upper[k]):
            return
        extent=upper[k]-lower[k]
        if extent>wp.float64(0.0):
            q[k]=wp.clamp((center[k]-lower[k])/extent,wp.float64(0.0),wp.float64(1.0))*wp.float64(2097151.0)
    keys[i]=expand_bits(wp.uint64(q[0])) | (expand_bits(wp.uint64(q[1]))<<wp.uint64(1)) | (expand_bits(wp.uint64(q[2]))<<wp.uint64(2))

'''
code=header;checks={}
for name in names:
 before=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);after=copy.deepcopy(before)
 assert ast.unparse(after.body[0])=='i = wp.tid()'
 after.name=name+'_ordered';after.args.args.append(ast.arg(arg='order',annotation=ast.parse('wp.array[int]',mode='eval').body));after.body[0]=ast.parse('i = order[wp.tid()]').body[0]
 normalized=copy.deepcopy(after);normalized.name=name;normalized.args.args.pop();normalized.body[0]=copy.deepcopy(before.body[0])
 assert ast.dump(normalized,include_attributes=False)==ast.dump(before,include_attributes=False)
 checks[name]=dict(only_added_order_argument_and_index_lookup=True,original_ast_sha256=hashlib.sha256(ast.dump(before,include_attributes=False).encode()).hexdigest())
 code+=ast.unparse(ast.fix_missing_locations(after))+'\n\n'
code+='''
ORDERED={original.classify_cached:classify_cached_ordered, original.selection_cached:selection_cached_ordered, original.retained_cached:retained_cached_ordered}

class OrderedSurfaceQueryCache(original.SurfaceQueryCache):
    def __init__(self,domain,capacity):
        super().__init__(domain,capacity)
        with wp.ScopedDevice(self.device):
            self.order_keys=wp.empty(2*capacity,dtype=wp.uint64)
            self.order_ids=wp.empty(2*capacity,dtype=int)
            self.order_enabled=wp.ones(1,dtype=int)
            wp.load_module(module=__name__,device=self.device,block_dim=32)
            # A child capture owns retained CUB temporary storage. A bare
            # top-level radix-sort capture may introduce graph allocations.
            with wp.ScopedCapture() as cap:
                wp.capture_if(self.order_enabled,self._sort)
            self.order_graph=cap.graph
        wp.synchronize_device(self.device)

    def _sort(self):
        radix_sort_pairs(self.order_keys,self.order_ids,self.capacity,end_bit=64)

    def launch(self,kernel,dim,args,block_dim=32):
        assert kernel in ORDERED and dim==self.capacity
        wp.launch(centroid_keys,self.capacity,inputs=[args[0],args[1],args[2],args[3],args[5],self.order_keys,self.order_ids],device=self.device,block_dim=256)
        wp.capture_launch(self.order_graph)
        wp.launch(ORDERED[kernel],dim,inputs=[*args,self.current,self.previous,self.table,self.order_ids],device=self.device,block_dim=block_dim)
        self.after(args[5])


def install():
    original.SurfaceQueryCache=OrderedSurfaceQueryCache
'''
p=w/'surface_query_order_v1.py';assert not p.exists();ast.parse(code);p.write_text(code)
out=w/'surface-query-order-source-audit-v1.json';assert not out.exists();out.write_text(json.dumps(dict(scope=__doc__,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kernel_ast_checks=checks,limitations=['Prototype is unexecuted: no compilation, numerical, capture, performance or memory qualification claimed.','Centroids are ordering keys only, not circumcenter replacements or geometric proofs.','Invalid inputs retain original classifiers; shared early-error flags require explicit multi-fault diagnostic tests.','Key generation, sorting, graph scratch and extra24*capacity array bytes must be included in end-to-end measurement.','Surface_error and other floating reductions are unchanged.']),indent=2)+'\n')
print('Prepared three scalar kernels with AST-equivalent bodies apart from index mapping; unexecuted prototype.')
