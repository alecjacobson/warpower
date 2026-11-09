# Interior witnesses for conservative containment

This unadopted experiment addresses an inconclusive case in material recovery
([#292](https://github.com/alecjacobson/warpower/issues/292)). The existing proof
can lack a strict vertex witness when all four tetrahedron vertices lie on input
triangle planes. The prototype leaves those original plane checks and vertex
queries intact and tries one rounded interior point only after they are
inconclusive. Production code does not import it.

The proof has the same prerequisites as the existing containment query: a valid
closed oriented input surface and a finite, nondegenerate, certified tetrahedron.
Every triangle whose axis-aligned bounding box intersects the tetrahedron box
must support all four vertices on its closed material side. Triangles rejected
by the box cannot intersect the tetrahedron. Because the tetrahedron is full
dimensional, its strict interior lies strictly on the accepted side of each
remaining triangle plane. Consequently, its interior cannot cross the input
surface, and one strict interior material witness certifies the whole cell's
closed containment.

The proposed point is the binary32 rounding of an arithmetic centroid computed
in binary64. Its construction is not the proof. Four existing filtered/exact
orientation predicates check the sub-tetrahedra obtained by replacing each
vertex with the proposed point. All must have the same nonzero orientation as
the original tetrahedron. This is precisely positivity of its four barycentric
coordinates. Any zero, opposite sign or unavailable predicate rejects the
witness. A representable strict interior point need not exist for a very thin
binary32 tetrahedron, and that case remains inconclusive. The original exact
material-side ray routine determines membership; unresolved rays never accept.

The diagnostic records support rejection, original vertex membership, absence of
a proved rounded interior witness, or use of the new interior query separately.
Its first scan check inspects unselected cells incident to invalid edges in the
saved rejected default Armadillo meshes. Those snapshots are diagnostic inputs,
not published valid meshes. No result is assumed: if their plane checks already
reject the cells, the extra witness will not fix those defaults.

Qualification results and performance are pending. This experiment neither
relaxes feature constraints nor changes which production cells are retained.
