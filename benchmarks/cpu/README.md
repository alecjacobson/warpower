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

The executable takes nine arguments including its program name:
`surface.off mesh.txt size facet_distance odt_iterations threads convergence freeze_bound`.
The sequential binary requires `threads=1`; use `threads=4` for the four-thread
comparison. The RNG seed is 2026. Feature detection uses 45 degrees, the facet
angle criterion is 25 degrees, and the cell radius-edge bound is 2. Perturbation
and exudation are disabled so the reported optimization is ODT; these settings
are not a claim that this is CGAL's best achievable quality or speed.

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

The output directory contains each native mesh, raw process output, and a
`result.json` with stage timings, hashes, parameters and independent quality.
Complete CPU time includes domain construction, feature protection, meshing
and ODT; file input, validity checking and file output are reported separately.
Independent boundary-distance measurements happen after all timing samples.
Their three-point quadrature RMS is not a Hausdorff certificate. This runner
does not itself establish quality parity or a GPU speedup.
