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

For at least 128 input sites, 32 lanes partition the hierarchy by the first five
binary path choices. An early leaf belongs only to the lane whose remaining
path bits are zero. Thus the starting subtrees are disjoint and cover every
leaf, even in unbalanced radix trees. Smaller inputs use one traversal per face.
A fixed worker pool with a device-counted strided tail covers all compacted
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

Validation records and the complete before/after benchmark matrix are being
prepared for integration. The initial complete 1,000-site torus measurement is
3,937 ms versus approximately 5,727 ms before this change, with identical
positions, material connectivity, counts and quality. These are resident GPU
pipeline timings, not a speedup over native CPU meshing. The native CPU goal
remains open.
