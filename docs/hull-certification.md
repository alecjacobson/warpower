# Hull-face certification

The optional certified Delaunay constructor verifies every exposed face against
all original sites. Together with the common symbolic lower-facet certificate
and opposite interior-face incidence, supporting boundary faces establish full
convex-hull coverage. This change preserves that acceptance rule.

The face hash first checks multiplicity and orientation, then compacts unpaired
face references into fixed storage. At most four faces exist per candidate
cell, so reserving `4 * tetrahedra_capacity` integers covers even malformed
candidates. One device counter delimits the prefix. This adds 16 bytes per
reserved tetrahedron and one integer counter, allocated during setup.

Each face caches its three normal coefficients as float64 value/error balls.
A node's box is translated relative to the first face vertex, and interval dot
products enclose its maximum plane value. A certified nonpositive bound skips
the node; uncertain nodes descend to the unchanged exact orientation predicate.
Reassociating the determinant through a cached cross product changes its error
bound, not the exact polynomial. Float32-coordinate cubic products and their
error bounds remain within the float64 normal range, including float32
subnormal inputs. Fast math and fused multiply-add remain disabled.

For each face, 32 lanes partition the hierarchy by the first five
binary path choices. An early leaf belongs only to the lane whose remaining
path bits are zero. Thus the starting subtrees are disjoint and cover every
leaf, even in unbalanced radix trees. Small trees simply leave unused lanes
inactive. The fixed worker pool has at least 128 face groups and at most 4,096,
which avoids concentrating tiny hulls into too few thread blocks. A
device-counted strided tail covers all compacted
faces without host count reads or dynamic allocations. Each subtree retains the
existing bounded traversal stack and failure reporting.

Only the canonical representative of coincident sites needs a plane test:
identical coordinates have exactly identical orientation signs. This avoids
repeating expensive exact zero tests on the duplicate positions used to mask
inactive sites during progressive meshing. Canonical ownership is rebuilt from
current coordinates on every reconstruction. Defining face vertices are also
skipped, as before.

The ordinary power-diagram, original Delaunay/regular-mesh routines, and their
capture APIs do not invoke this optional certificate and retain their code.
The certificate's public inputs, outputs and error bits are unchanged. Boundary
scratch is temporary; consumers must use published mesh outputs, particularly
after optimizer rollback.

## Validation and performance

The [374-test suite](../results/validation/vtm/hull-certificate-final-tests.txt)
passes on the final launch configuration. New cases check 640 plane/box
configurations against rational arithmetic across float32 scales and offsets,
capacity-one storage of all four exposed faces, and missing-cell rejection on
an unbalanced radix tree with 192 coincident sites. Existing tests cover corrupt
candidates, cospherical ties, changing inputs and complete meshing trajectories.

Five [padded memory cases](../results/validation/vtm/hull-certificate-memcheck.txt)
and one [changing-input lifetime case](../results/validation/vtm/hull-certificate-lifetime.txt)
pass with zero errors. Diagnostic checks use a 32 KiB stack; production settings
are unchanged. The [native full-mesher trace](../results/validation/vtm/hull-certificate-device-audit.json)
records 83,649 kernel activities over three complete replays. Its graph has 6,700
kernel, 2,014 memset, 770 conditional and 408 device-copy nodes, with no host or
allocation/free nodes. Every observed copy and graph copy node is device-to-device;
one final synchronization is outside geometry.

The [100,000-site reference comparison](../results/validation/vtm/hull-certificate-100k-cgal.json)
matches all 671,908 tetrahedra against saved native CGAL 6.2.1 output and the
previous certified constructor. This is independent connectivity validation,
not a CPU timing comparison.

The [paired timing matrix](../results/validation/vtm/hull-certificate-performance.json)
uses CUDA events on an NVIDIA L40, Warp 1.19.0, with no observed CPU-quota
throttling. Setup, JIT and uploads are excluded. Stages are measured separately,
so their medians need not add to the complete median.

| Input | Complete before | Complete after | Certificate before | Certificate after |
|---|---:|---:|---:|---:|
| 8 cube corners | 0.529 ms | 0.524 ms | 0.154 ms | 0.149 ms |
| 64-site lattice | 2.338 ms | 2.211 ms | 0.702 ms | 0.582 ms |
| 200-site corner cloud | 8.464 ms | 8.360 ms | 3.083 ms | 2.938 ms |
| 128 structured torus sites | 1,228.65 ms | 1,220.99 ms | 91.189 ms | 90.947 ms |
| 1,000 random sites | 12.536 ms | 10.240 ms | 3.642 ms | 1.308 ms |
| 100,000 random sites | 765.06 ms | 563.22 ms | 291.26 ms | 89.97 ms |

The [complete 1,000-site torus mesher](../results/validation/vtm/hull-certificate-torus1000.json)
falls from [5,726.57 ms](../results/validation/vtm/symbolic-cached-full-torus1000.json)
to 3,937.38 ms (1.45×), including domain refit, fresh sampling, growth, fitting,
quality rounds and extraction. Both measurements use three warmed samples and
unthrottled CPU windows. Point positions and canonical full/material cells are
[identical](../results/validation/vtm/hull-certificate-equivalence.json), with
1,000 sites, 5,826 full cells, 4,225 retained cells and 1,122 boundary triangles.
Minimum radius ratio remains 0.092858, median 0.874421, relative volume error
−5.089%, and directed surface RMS 0.019723. These quality limits are unchanged.
The [isolation record](../results/validation/vtm/hull-certificate-isolation.json)
checks every other runtime module against the preceding main revision.

These are comparisons against the preceding Warpower implementation, not against
native CPU meshing. The full CPU speedup goal remains open. The large random
case is now dominated by candidate construction, tracked in
[#87](https://github.com/alecjacobson/warpower/issues/87).
