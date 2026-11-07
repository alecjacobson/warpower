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
contribute. The measured validation follows.

## Measured results

The [comparison record](../results/validation/vtm/containment-summary.json) and
[artifact manifest](../results/validation/vtm/containment-provenance.json) retain
raw timings, saved meshes, independent checks and source hashes. Measurements
use an NVIDIA L40 and Warp 1.19, comparing `fa59aa8` with runtime `e85d086`.
Each sample is a complete fresh resident solve including domain/sizing refits
and initial sampling. Setup, transfers, export and independent checking are
outside replay timing. All recorded replay windows have zero observed CPU quota
throttling.

| Previously valid case | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Torus, 1,000 sites | 2,227.789 | 2,227.751 |
| Torus, 4,096 sites | 8,620.533 | 8,626.763 |
| Graded torus, 1,000 sites | 3,716.654 | 3,723.822 |
| Thin hand, 1,000 sites, quadrature 20/20 | 3,364.006 | 3,365.021 |
| Cavity, 1,000 sites, quadrature 20/20 | 1,419.612 | 1,420.259 |
| Disconnected, 1,000 sites, quadrature 20/20 | 1,240.688 | 1,243.919 |

These are medians of three warm replays. All differences are below 0.3%.
Cell sets, oriented boundaries and roles agree in all six cases. The hand has
small coordinate differences; the other five have bitwise-identical positions.
Setup storage increases: the measured device pool changes from 264.005 to
272.603 MB at capacity 1,000 and from 1,670.773 to 1,705.410 MB at capacity 4,096.
Corresponding setup times change from 1.230 to 1.300 s and 2.481 to 2.517 s.

Standalone Delaunay medians change from 9.183 to 9.169 ms at 1,000 sites and
182.796 to 182.111 ms at 100,000 sites. The 100,000-site diagram changes from
18.201 to 18.123 ms. Their implementations and the README are unchanged;
these small timing differences are not attributed to the new meshing feature.
CPU outperformance remains unachieved.

Both coarse examples below use quadrature 4/4, 1,000 sites, 256 initial sites
and four quality rounds. Recovery avoids additional global fitting in these
trajectories; it does not establish convergence or a quality threshold.

| Measurement | Coarse cavity | Coarse disconnected |
| --- | ---: | ---: |
| Full-hull / material tetrahedra | 5,220 / 4,545 | 5,826 / 4,350 |
| Boundary triangles | 1,040 | 1,058 |
| Material components / boundary shells / cavities | 1 / 2 / 1 | 2 / 2 / 0 |
| Minimum / mean radius ratio | 0.09908 / 0.81932 | 0.06431 / 0.81209 |
| Two-sided sampled RMS / input box diagonal | 0.47447% | 0.38349% |
| Relative volume error | −2.4191% | −2.3415% |
| Complete resident median of three warm replays | 3.038 s | 3.141 s |

The preceding revision rejects the coarse cavity. The previous
[coarse disconnected record](extraction-geometry.md) took 11.743 s, with minimum
radius ratio 0.06904 and sampled RMS error 0.33674%. The shorter new trajectory
has slightly worse quality and boundary fit, so this is not a quality-matched
speedup claim. Fewer recovery fitting rounds can change the result at the same
maximum budgets; disabling containment recovery retains the preceding policy.
The dense 20/20 gallery configurations remain preferable for these examples.

All eight successful final full-hull meshes pass independent CGAL 6.2.1 exact
orientation, global empty-sphere, face-pairing and hull-support checks. NumPy
checks material quality and topology, and VTU round trips match the saved arrays.
The coarse thin hand still fails with error 64 and zero public mesh counts;
this conservative repair adds no cells there. Its remaining work is tracked in
[#112](https://github.com/alecjacobson/warpower/issues/112).

The complete suite passes 528 tests; the subsequently added small nested
regression also passes. Eight isolated regressions and one complete repaired
128-site graded quality trajectory pass unfiltered padded memory checking with
zero errors. The four-phase resident audit passes unfiltered allocation/lifetime
checking. Its native trace records 484 kernel activities, 72 memsets, four graph
launches and one final synchronization. The graph contains 112 kernel nodes,
70 memset nodes, six conditionals and two device-to-device copies; it contains
no host copies, callbacks, allocation or free nodes.

The complete current 1,000-site disconnected trajectory also passes unfiltered
padded memory checking: one test passes in 175.92 s with zero errors, within a
six-minute observation budget. The [scope record](../results/validation/vtm/containment-full-disconnected-audit.json)
and [complete log](../results/validation/vtm/containment-full-disconnected-memcheck.txt)
identify the runtime, fixture and diagnostic configuration. This completes
[#129](https://github.com/alecjacobson/warpower/issues/129) for the current
implementation. The earlier `1804b64` attempts were stopped before completion
and remain incomplete; the new trajectory's pass does not retroactively validate
those canceled runs or establish whether they would eventually have finished.
