"""Count fast-filter box decisions while retaining the original traversal and decisions."""
import warp as wp
from warpower import _proved_completion
from warpower import _bounded_predicates as af
from warpower._bvh import Tree
from warpower._delaunay_certificate import halfspace,inside_halfspace
from warpower._mesh_topology import face_vertices
from warpower._proved_completion import axis_supported
from bounded_hull_box_v1 import maximum_corner,bounded_inside
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
instances=[]
COLUMNS=['calls','partition_face_visits','axis_partition_accepts','visited_nodes','accepted_boxes','visited_leaves','site_slots','site_queries','bounded_face_and_node','overflow_returns','mask_available_nodes','fast_boxes','fast_and_old_boxes','fast_only_boxes']
@wp.kernel
def record_face_support(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    boundary: wp.array[int],
    count: wp.array[int],
    tree: Tree,
    error: wp.array[int],
    depth: int,
    workers: int,
    supported: wp.array[int],
    stats: wp.array[wp.int64],
):
    tid = wp.tid()
    if tid == 0:
        wp.atomic_add(stats, 0, wp.int64(1))
    face_work, axis_work, node_work, box_work = int(0), int(0), int(0), int(0)
    leaf_work, slot_work, query_work, bounded_nodes = int(0), int(0), int(0), int(0)
    mask_nodes, fast_boxes, both_boxes, fast_only = int(0), int(0), int(0), int(0)
    pieces = 1 << depth
    worker, lane = tid // pieces, tid % pieces
    # Partition the radix tree by path bits. An early leaf belongs only to
    # the lane whose unused bits are zero, so every subtree is covered once.
    root = int(1)
    live = bool(True)
    for level in range(depth):
        if tree.count[root] > 0:
            live = (lane >> level) == 0
            break
        if ((lane >> level) & 1) == 0:
            root = tree.left[root]
        else:
            root = tree.right[root]
    if not live:
        return
    # Fixed launch size, device count, and a strided tail cover even malformed
    # candidates exposing all four faces per tetrahedron.
    for slot in range(worker, count[0], workers):
        # A different partition may already have disproved this same face.
        # Atomic reads match the monotone atomic clears below.
        if wp.atomic_add(supported, slot, 0) == 0:
            continue
        face_work += 1
        ids = face_vertices(tets, boundary[slot])
        a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
        face = boundary[slot]
        opposite = points[tets[face // 4, face % 4]]
        if axis_supported(a, b, c, opposite, tree.lower[1], tree.upper[1]):
            axis_work += 1
            continue
        finite_face = af.bounded(a) and af.bounded(b) and af.bounded(c)
        plane = halfspace(a, b, c)
        mask = maximum_corner(plane)
        stack = wp.zeros(shape=128, dtype=int)
        stack[0] = root
        size = int(1)
        while size > 0:
            if wp.atomic_add(supported, slot, 0) == 0:
                break
            size -= 1
            node = stack[size]
            node_work += 1
            if finite_face and af.bounded(tree.lower[node]) and af.bounded(tree.upper[node]):
                bounded_nodes += 1
            old = inside_halfspace(plane, tree.lower[node], tree.upper[node])
            fast = bounded_inside(a,b,c,mask,tree.lower[node],tree.upper[node])
            mask_nodes += int(mask>=0)
            fast_boxes += int(fast)
            both_boxes += int(fast and old)
            fast_only += int(fast and not old)
            if old:
                box_work += 1
                continue
            if tree.count[node] > 0:
                leaf_work += 1
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    slot_work += 1
                    q = tree.ids[k]
                    # Coincident sites have exactly the same plane sign.
                    if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2]:
                        continue
                    query_work += 1
                    result = af.orient3d(a, b, c, points[q])
                    if result.valid == 0:
                        wp.atomic_or(error, 0, 2)
                    elif result.sign > 0:
                        wp.atomic_or(error, 0, 64)
                        wp.atomic_and(supported, slot, 0)
                        # One exact exterior witness disproves this face's
                        # support. End only this subtree/face, not the thread:
                        # the strided outer loop must still check later slots.
                        size = 0
                        break
            else:
                if size > 125:
                    wp.atomic_or(error, 0, 16)
                    wp.atomic_add(stats, 9, wp.int64(1))
                    return
                stack[size] = tree.left[node]
                stack[size + 1] = tree.right[node]
                size += 2
    wp.atomic_add(stats, 1, wp.int64(face_work))
    wp.atomic_add(stats, 2, wp.int64(axis_work))
    wp.atomic_add(stats, 3, wp.int64(node_work))
    wp.atomic_add(stats, 4, wp.int64(box_work))
    wp.atomic_add(stats, 5, wp.int64(leaf_work))
    wp.atomic_add(stats, 6, wp.int64(slot_work))
    wp.atomic_add(stats, 7, wp.int64(query_work))
    wp.atomic_add(stats, 8, wp.int64(bounded_nodes))


    wp.atomic_add(stats, 10, wp.int64(mask_nodes))
    wp.atomic_add(stats, 11, wp.int64(fast_boxes))
    wp.atomic_add(stats, 12, wp.int64(both_boxes))
    wp.atomic_add(stats, 13, wp.int64(fast_only))

def install():
 cls=_proved_completion.ProvedCertificate;original_init=cls.__init__
 def init(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.hull_box_stats=wp.zeros(len(COLUMNS),dtype=wp.int64)
   wp.load_module(module=__name__,device=self.device,block_dim=32)
  instances.append(self)
 def hull(self,tree):
  self._face_supported.fill_(1)
  self._launch(record_face_support,self._face_workers*(1<<self._face_depth),[self.points,self.tetrahedra,self._boundary,self._boundary_count,tree,self.error,self._face_depth,self._face_workers,self._face_supported,self.hull_box_stats])
 cls.__init__=init;cls._hull_support=hull
 return instances
