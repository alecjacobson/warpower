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
