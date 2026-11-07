# Conservative material-boundary recovery

The circumcenter extraction rule can omit a tetrahedron that lies entirely
inside material. In the coarse cavity example, two such omissions break the
boundary fan around an edge. Delaunay certification still passes, but the
material subcomplex is nonmanifold.

`ExtractionWorkspace(..., containment_recovery=True)` enables a bounded device
repair. The independent extractor defaults to `False`, preserving the paper's
circumcenter rule exactly. The full `VariationalMeshingWorkspace` enables it by
default; pass `containment_recovery=False` to use only the original extraction
and additional-fitting recovery policy. This extension does not change the
independently callable diagram or Delaunay routines.

## Which cells can be added

Recovery starts only when classification succeeds and the extracted boundary
has an edge-incidence diagnostic. It examines excluded input tetrahedra touching
those bad edges, using the existing boundary-edge hash table. It adds a cell
only when a conservative geometric test proves that the entire cell lies in
the closed material:

1. Compute the tetrahedron's exact float32 axis-aligned bounding box. For every
   input triangle with an overlapping box, require every tetrahedron vertex to
   lie on or inside that triangle's outward plane. Original-coordinate robust
   orientation predicates resolve uncertain signs.
2. Find an original tetrahedron vertex strictly off all those planes. Triangles
   with disjoint boxes cannot contain it, so this vertex is off the entire
   surface. The plane conditions prevent any input triangle from crossing the
   tetrahedron interior.
3. Establish material membership of this strict witness using exact integer
   ray winding. Strict triangle-interior crossings contribute their oriented
   signs. Rays hitting an edge or lying in a triangle plane are inconclusive;
   retry up to three fixed directions. Accept only an unambiguous winding of
   one. Arithmetic failure or exhausted witnesses leave the cell excluded.

The membership calculation uses bounded integer arithmetic on the original
float32 coordinates. No perturbed site, rounded centroid, CPU query or host
decision is substituted. The test is sufficient, not necessary: safe cells can
fail its conservative plane or witness requirements. Testing only the four
vertices for material membership would be insufficient; a tetrahedron can have
all vertices in material while crossing an enclosed cavity.

The input domain must already pass surface validation and the input tetrahedra
must form a nonoverlapping, positively oriented complex. The extractor does not
independently certify arbitrary input connectivity. Adding a subset of an
already-certified full-hull Delaunay complex preserves that prerequisite.

## Bounded replay and quality guards

Each pass adds proven-contained cells, compacts the selection and reruns the
complete topology diagnostics. There are at most four passes. No additions,
resolved topology or an unsupported diagnostic terminates recovery. Remaining
failure is explicit; the full mesher still requires its topology to match the
input and otherwise uses its existing fitting/rejection policy.

With recovery enabled, the extractor exposes device `repaired_count` and
`repair_rounds` scalars. They reset each replay. `selected` and `source_indices`
describe the final selection, including additions; circumcenter and signed
distance arrays still describe the original geometric measurements. Repair
does not move sites, create connectivity or remove previously retained cells.

Once a full meshing trajectory uses repair, its selective quality guards evaluate
the same repaired extraction. Jitter and interior proposals likewise assess the
complete repaired subcomplex. The first selective stage that activates repair
also checks its final minimum quality against its checkpoint, with rollback on
regression. A fresh solve clears this trajectory flag. Ordinary valid trajectories
keep the cheaper quality-classification path.

All buffers and conditional graphs are constructed at setup. Replay performs no
allocation or host readback. The rare containment calculation currently scans
input triangles; it is not a general accelerated exact inside/outside API.
Recovery does not establish a minimum element quality, convergence, exact input
facet conformity or a maximum boundary-distance guarantee.

## Regression coverage

The small saved five-tetrahedron cavity fan retains three cells under ordinary
classification. Recovery adds exactly its two wholly material omissions and
produces a closed manifold boundary. Adversarial tests reject cavity-crossing
cells, exterior cells and bridges between disconnected components. Exact
Fraction Gaussian solves check winding and ambiguous corner rays at coordinate
scales `2^-100`, `1` and `2^100`.

Captured producer/consumer tests cover repaired, nonfinite, empty and restored
frames, including invalid inactive padding. A separate quality-selection test
switches the device recovery flag and verifies that all five repaired cells
contribute. Full-trajectory and performance validation records will be linked
with the implementation's measured results before merge.
