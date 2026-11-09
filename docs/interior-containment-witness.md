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

## Result: no recovery of the retained Armadillo candidates

Tested source `15bfb42` passes eleven focused tests, including seven new
experimental cases. The proposed interior witnesses agree with exact rational
barycentric sign checks on 164 ordinary, flattened, extreme-scale, subnormal,
one-ULP and coplanar tetrahedra. The boundary-only primitive succeeds at scales
2^-60, 1 and 2^60, while cavity and crossing counterexamples remain rejected.
The captured producer/consumer case changes current points, passes an empty
active prefix containing invalid padding, and recovers. All seven new cases
also pass isolated padded and full allocation-order memory checks with zero
reported errors.

| Rejected snapshot | Sites | Unselected cells incident to bad edges | Rejected by support planes | Newly certified cells |
|---|---:|---:|---:|---:|
| Default Armadillo | 1,000 | 98 | 98 | 0 |
| Denser Armadillo | 8,192 | 33 | 33 | 0 |

These are retained rejected candidates from the earlier initial-budget study.
Both have independently certified full hulls, but invalid material boundaries.
The new witness is never reached in any of the 131 examined cells. This does not
prove those cells exterior: a failed support-plane test remains inconclusive.
It does show that this specific fallback cannot recover these selections.
No production adoption or complete-solve speed claim follows from the experiment.
The default workflow remains open in #263.

Two earlier attempts are retained. The first used an unsupported tuple assignment
to Warp array elements and failed compilation (seven new tests failed; four old
controls passed). The second passed ten tests but failed the smallest-scale
primitive because the generic CPU fixture helper oriented faces with an
underflowing binary32 cross/dot calculation. An independent binary64 check found
six inward faces. The corrected fixture orients the unit cube before scaling,
checks domain validity explicitly, and has twelve outward faces. The production
geometry predicates and common fixture helper were not changed.

Compact records and checksums are in the
[evidence index](../experiments/interior-containment-witness/). Runtime source,
public APIs and the flagship README are unchanged.
