# Independent native CPU reference

`delaunay.cpp` constructs an ordinary full-hull Delaunay mesh with CGAL's exact
predicates and validates the result. It is a benchmark/validation tool, not a
Warpower runtime dependency. The production package uses Warp kernels.

The validated reference version is [CGAL 6.2.1](https://github.com/CGAL/cgal/releases/tag/v6.2.1).
Install a C++17 compiler, Boost, GMP, and MPFR development packages. With the CGAL
library release extracted locally:

```bash
mkdir -p results/local
g++ -O3 -DNDEBUG -std=c++17 -I/path/to/CGAL-6.2.1/include \
  benchmarks/cpu/delaunay.cpp -lgmp -lmpfr -o results/local/cgal-delaunay
```

Input is a text file beginning with the point count, followed by one `x y z` row
per point. Output begins with the tetrahedron count, followed by four zero-based
input indices per row. The executable accepts input/output paths as its two
arguments and writes version, counts, and construction seconds as JSON to stderr.
Timing excludes input/output and the independent validity check; it is not a
complete meshing-pipeline CPU benchmark. This reference uses sequential CGAL
Delaunay insertion. Full domain-meshing and parallel CPU comparisons are separate
work tracked in issue #40.

For an optimized saved dataset:

```bash
python -m benchmarks.odt --sizes 1000 10000 --iterations 20 --reference \
  --save-dir results/local/odt --output results/local/odt-reference.json
```

Convert a saved dataset to the reference's input format:

```python
import numpy as np

points = np.load("results/local/odt/optimized-10000.npz")["points"]
with open("results/local/points.txt", "w") as f:
    f.write(f"{len(points)}\n")
    np.savetxt(f, points, fmt="%.17g")
```

Then run `results/local/cgal-delaunay results/local/points.txt results/local/tets.txt`.
Compare tetrahedron sets after sorting each quadruple and sorting rows; ordering
and orientations can differ between implementations. Degenerate inputs can have
multiple valid Delaunay triangulations, so exact set equality alone is not a
suitable correctness criterion for those cases.

`check_delaunay.cpp` independently checks a supplied full-hull triangulation
without requiring CGAL to choose the same cospherical diagonals. Compile it with:

```bash
g++ -O3 -std=c++17 -frounding-math -I/path/to/CGAL-6.2.1/include \
  benchmarks/cpu/check_delaunay.cpp -lgmp -lmpfr -o results/local/check-delaunay
results/local/check-delaunay results/local/points.txt results/local/tets.txt
```

The two inputs use the point and tetrahedron formats described above. Use
`full_tetrahedra` from a gallery NPZ export: the nonconvex material subset is
not the full convex hull. On the original coordinates, CGAL's exact predicates
check positive cell orientation, every site's exclusion from each open
circumsphere (cospherical ties are allowed), paired-face orientation, and
supporting hull planes. The program also checks cell uniqueness, distinct-site
coverage and hull-edge incidence. It prints JSON on success and exits nonzero
on malformed input or failed checks. This deliberately exhaustive O(sites ×
cells) check is an offline validator for small examples, not a meshing benchmark
or part of the device pipeline. The
[thin-hand check record](../../results/examples/vtm/thin-hand/export-and-native-check.json)
contains the full-hull result and positive/negative fixture inputs and outcomes.

`surface_intersections.cpp` and `triangle_contact.cpp` are additional independent
CGAL 6.2.1 references for [surface-contact validation](../../docs/domains.md).
Both use `Exact_predicates_exact_constructions_kernel`. Compile with the same
include path, C++17 and GMP/MPFR libraries as above. They are host-only validation
programs and are never called by the device pipeline.

The surface program takes a triangular OFF path and prints vertex, face and
self-intersection counts, followed by at most 256 face-pair witnesses. Convert
input positions to float32 and print them with 17 significant digits first;
the reader explicitly constructs exact points from doubles so the reference
uses the same binary coordinates as the GPU. The pair program reads a count,
then six vertex IDs and six XYZ triples per pair from stdin. Each output row
reports the row, shared-index count, whether an exact intersection exists,
whether it exceeds legal indexed contact, and the CGAL variant index (`-1` for
empty). The [pinned input URLs and hashes](../../results/validation/vtm/domain-contact-inputs.json)
and [whole-surface exact results](../../results/validation/vtm/domain-contact-CGAL.json)
record the imported-model check. The regression suite additionally proves
separation independently with Python arbitrary-precision integers.

## Complete domain meshing and ODT

`mesh3.cpp` builds a feature-aware polyhedral domain, runs native Mesh_3, then
optionally runs CGAL's ODT optimizer. It is independent of Warpower geometry.
CGAL 6.2.1, Eigen 3, and (for the parallel build) TBB development headers are
required in addition to the dependencies above. Example release builds:

```bash
g++ -O3 -DNDEBUG -std=c++17 -I/path/to/CGAL-6.2.1/include -I/usr/include/eigen3 \
  benchmarks/cpu/mesh3.cpp -lgmp -lmpfr -o results/local/cgal-mesh3-sequential
g++ -O3 -DNDEBUG -std=c++17 -DWARPOWER_CGAL_PARALLEL -DCGAL_LINKED_WITH_TBB \
  -I/path/to/CGAL-6.2.1/include -I/usr/include/eigen3 benchmarks/cpu/mesh3.cpp \
  -lgmp -lmpfr -ltbb -ltbbmalloc -o results/local/cgal-mesh3-parallel
```

The executable takes eight required arguments after its program name:
`surface.off mesh.txt size facet_distance odt_iterations threads convergence freeze_bound`,
followed optionally by `feature_angle [non_manifold|manifold]`.
The sequential binary requires `threads=1`; use `threads=4` for the four-thread
comparison. The RNG seed is 2026. Feature detection defaults to 45 degrees, the facet
angle criterion is 25 degrees, and the cell radius-edge bound is 2. By default,
perturbation and exudation are disabled so the reported optimization is ODT; these settings
are not a claim that this is CGAL's best achievable quality or speed.

The Python runner exposes `--feature-angle` and `--topology`. The defaults retain
the historical 45-degree, `non_manifold` policy and command compatibility with
older binaries. An explicit `manifold` request uses CGAL's closed-manifold
refinement option. Native triangulation validity alone does not establish a valid
material boundary, and an initially manifold result still needs validation after
ODT. Equal feature angles also do not establish equivalent GPU/CPU corner or
boundary-update constraints. See the [CGAL topology options](https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html).

Convergence 0.02 and freeze bound 0.01 match the shipped CGAL 6.2.1 ODT stopping
defaults. Zero disables the respective early stop for a separately labeled
fixed-iteration experiment. Report these parameters with every result. The
CPU and GPU algorithms use different boundary constraints and stopping rules;
compare measured quality and boundary error as well as counts and time.

The Python runner checks that input coordinates round-trip exactly through
float32, invokes fresh native processes, records CPU quota observations, then
independently measures every distinct output using NumPy geometry. Use the
same exported input surface as the GPU experiment:

```bash
python -m benchmarks.cpu_mesher --binary results/local/cgal-mesh3-sequential \
  --surface results/local/torus/input.off --size 0.32 --facet-distance 0.025 \
  --odt-iterations 50 --convergence 0.02 --freeze-bound 0.01 --threads 1 \
  --repeats 3 --output results/local/cpu-torus
```

For dense scanned surfaces, add `--distance-binary results/local/surface-distance`
to use the optional native AABB oracle described below during offline validation.
Its binary/source hashes are recorded; it runs after the native timing samples.

The output directory must be empty. It contains each native mesh, raw process
output, a progressively saved `runs.json`, and a `result.json` with stage timings,
hashes, parameters and independent quality. Native timing and quota records are
saved before quality validation, so a failed check does not discard them.
The runner exits nonzero if any repeat fails. `native_valid` and `quality_valid`
are separate: the latter means the independent material metric routine passed,
not that every possible geometric property has been certified. Studies also
check boundary vertex links and exact self-intersections where claimed.

`median_complete_seconds` and `median_unthrottled_complete_seconds` use only
individually successful samples; they are null when none qualify. A mixed batch
can have a successful-sample median while its overall `valid` flag is false.
`median_native_complete_seconds` retains native-process observations even when
their material meshes fail validation; it is not an accepted meshing result.
Complete CPU time includes domain construction, feature protection, meshing
and ODT; file input, validity checking and file output are reported separately.
Independent boundary-distance measurements happen after all timing samples.
Their three-point quadrature RMS is not a Hausdorff certificate. This runner
does not itself establish quality parity or a GPU speedup.

## Offline triangle-distance oracle

`surface_distance.cpp` supplies CGAL AABB closest-triangle distances for dense
surface validation. It is independent of the pure-Warp mesher and is called only
by optional offline benchmark utilities. Build and validate it with:

```bash
g++ -O3 -std=c++17 -frounding-math -I/path/to/CGAL-6.2.1/include \
  benchmarks/cpu/surface_distance.cpp -lgmp -lmpfr -o results/local/surface-distance
python -m benchmarks.surface_distance_audit --binary results/local/surface-distance \
  --scratch results/local/distance-scratch --output results/local/distance-audit.json
```

The executable accepts `target.off queries.txt`. The target is a nonempty
triangular ASCII OFF mesh; queries start with a count followed by XYZ rows.
It writes one squared distance per query and rejects malformed input or
zero-area target triangles. `--version` prints its CGAL version. Coordinates and
distance constructions are float64; this is not an exact Hausdorff checker.

`benchmarks.mesh_quality --distance-binary results/local/surface-distance` uses
this oracle for both sampled boundary-distance directions. Topology, volume,
radius ratios and dihedral angles still use the independent NumPy routines.
Temporary input files stay under the report's `distance-scratch` directory and
are removed after each call; final reports record source and executable hashes.
The default quality checker continues to use brute-force triangle distances.
See the [scanned Armadillo example](../../docs/scanned-armadillo.md).

## Optional quality optimizers

New binaries additionally accept six trailing arguments, after an explicit
feature angle and topology:
`perturb_enabled perturb_seconds perturb_bound exude_enabled exude_seconds exude_bound`.
Each enable flag is 0 or 1; time limits are finite and nonnegative; angle targets
are in [0,180]. Zero time means no optimizer time limit. The Python switches are
`--perturb`, `--perturb-seconds`, `--perturb-bound`, and the analogous `--exude`
options. Both optimizers stay disabled by default. When enabled, this benchmark's
default limit is one second and target is ten degrees; these are benchmark
settings, not a claim about CGAL's optimal configuration.

The order is ODT, perturbation, then exudation. Perturbation moves vertices;
exudation changes weights and must run last. See the
[CGAL optimizer reference](https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Functions.html).
All phase times and return codes are recorded. An iteration count is a requested
limit, not an observed count. `prepared_domain_seconds` measures meshing and all
optimizers after domain preparation; `complete_seconds` additionally includes
that preparation. Neither includes input, validation, or output.

Each new native run exports `mesh-N.txt.weights`, with the exported vertex count
then one weight per vertex in the same order as its coordinate file. The runner
validates and hashes this sidecar. The triangulation is labeled `weighted_regular`;
feature protection can already give nonzero weights before exudation. Independent
material-quality checks do not certify an ordinary unweighted Delaunay mesh.
The coordinate/tet file remains compatible with the historical format.

The runner sends the original positional command when both optimizers are off,
so older executables remain usable. `--timeout` bounds each native process
(default 600 seconds); timed-out runs retain partial stdout/stderr and failure
records, are excluded from successful timing medians, and cause a nonzero runner
exit. Native optimizer time limits and this process timeout have different scopes.

The [qualified optimizer study](../../docs/cpu-quality-optimizers.md) retains
complete/prepared-domain timings, quality tradeoffs and rejected controls.
