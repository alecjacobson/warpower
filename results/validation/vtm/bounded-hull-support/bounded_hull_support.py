"""Private bounded orientation in hull-support leaf queries only."""
import warp as wp
from warpower import _bounded_predicates as af
from warpower import _filtered_predicates as fp
from warpower._bvh import Tree
from warpower._mesh_topology import face_vertices
from warpower._delaunay_certificate import halfspace, inside_halfspace
import warpower._proved_completion as pc

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

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
):
    tid = wp.tid()
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
        ids = face_vertices(tets, boundary[slot])
        a, b, c = points[ids[0]], points[ids[1]], points[ids[2]]
        plane = halfspace(a, b, c)
        stack = wp.zeros(shape=128, dtype=int)
        stack[0] = root
        size = int(1)
        while size > 0:
            if wp.atomic_add(supported, slot, 0) == 0:
                break
            size -= 1
            node = stack[size]
            if inside_halfspace(plane, tree.lower[node], tree.upper[node]):
                continue
            if tree.count[node] > 0:
                for k in range(tree.start[node], tree.start[node] + tree.count[node]):
                    q = tree.ids[k]
                    # Coincident sites have exactly the same plane sign.
                    if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2]:
                        continue
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
                    return
                stack[size] = tree.left[node]
                stack[size + 1] = tree.right[node]
                size += 2



def install():
    pc.record_face_support=record_face_support
    original_init=pc.ProvedCertificate.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        wp.load_module(module=__name__,device=self.device)
    pc.ProvedCertificate.__init__=init
