# Exact surface crossings in default corner recovery

The default 45-degree Armadillo configuration still fails material-boundary
validation ([#263](https://github.com/alecjacobson/warpower/issues/263)). Its
retained rejected 1K and 8K candidates have independently certified full
Delaunay hulls. The earlier [interior-witness experiment](interior-containment-witness.md)
found that all 131 unselected cells incident to invalid boundary edges failed
the conservative triangle-plane support test. A failed sufficient test does
not establish that a cell crosses the actual surface.

This offline experiment ([#294](https://github.com/alecjacobson/warpower/issues/294))
separates actual crossings, boundary-only contact and separation. It changes
no runtime, public API, default or production dependency. CPU exact arithmetic
is used only for independent diagnosis, never in the device meshing loop.

## Results

| Rejected snapshot | Unselected candidates | Strict surface crossings | Wholly exterior | Wholly material | Bad boundary edges after safe additions |
|---|---:|---:|---:|---:|---:|
| Default Armadillo, 1K sites | 98 | 77 | 19 | 2 | 29 → 28 |
| Default Armadillo, 8K sites | 33 | 22 | 10 | 1 | 10 → 10 |

The two material cells at 1K are full-hull rows 1452 and 5271; the one at 8K
is row 34190. These indices apply only to the hashed snapshots. Adding all
three proved cells does **not** produce a valid material mesh. The 1K selection
still omits two mandatory corner sites; the 8K selection still omits one.
The other 99 candidates intersect the input surface in their strict interior,
and therefore cannot be accepted by a proof of whole-cell containment.
The 29 noncrossing exterior cells must not be added as material.

Stronger containment would accept a few legitimate cells, but it cannot by
itself repair these edge stars. No production containment replacement is
adopted, and neither rejected snapshot is a usable output. The default issue
remains open for site/feature placement and material recovery. These results
do not justify dropping corners, weakening topology validation, or claiming a
performance gain.

## Exact method and independent checks

All stored binary32 coordinates are converted to exact rational numbers.
Each input triangle whose closed coordinate box overlaps a tetrahedron's box
is clipped against its four closed barycentric half-spaces. Rational
interpolation carries both tet and triangle barycentric weights. This computes
the closed intersection, including coplanar faces, edges and vertices.

The clipped polygon intersects the **strict** tet interior exactly when no tet
barycentric coordinate is identically zero on it. Every coordinate is
nonnegative throughout the clipped polygon, so the average of its vertices
has all four coordinates positive precisely in that case. For every reported
crossing, a triangle-barycentric certificate reconstructs a point on the
triangle, and four separate exact determinants verify strict tet membership.
No rounded centroid is used to certify intersections.

CGAL 6.2.1's exact-predicates/exact-constructions triangle–tetrahedron intersection
independently agrees on both closed contact and strict intersection for all
**17,919** overlapping-box pairs: 16,026 at 1K and 1,893 at 8K. It also agrees
on **560** analytic/random controls, including coplanar contacts, shared edges,
thin cells, winding permutations and extreme scales. The rational tool's
**27 tests** pass, including subnormal and one-ULP tetrahedra.

For each of the 32 noncrossing cells, a separate exact CGAL side-of-mesh query
classifies its exact arithmetic centroid. The centroid is strictly inside the
nondegenerate tet. Because the tet interior is connected and has no surface
intersection, this single query determines its entire interior's material
side. This reasoning requires the validated, closed input surface; it is not
a centroid-only containment heuristic for crossing cells.

Safe-addition results are offline incidence diagnostics. Remaining bad edges
already disqualify both candidates; no complete-mesh acceptance or timing is
claimed. Production code and the flagship README are byte-identical to the
base revision, so no GPU performance or capture claim follows from this CPU
experiment.

## Reproduce

The evidence bundle includes both exact-coordinate snapshots, full rational
certificates, CGAL pair inputs/outputs, side-query inputs/outputs, source pins
and executable driver commands. The initial diagnostic source is `9780ffc`;
the material-side oracle and import-format cleanup are at `6190cc3`.

```bash
python -m pytest -q tests/test_corner_intersections.py
python -m benchmarks.corner_intersections \
  --input rejected-candidate.npz --output intersections.json \
  --reference-input intersections.txt
c++ -O2 -DNDEBUG -std=c++17 -I/path/to/CGAL/include \
  benchmarks/cpu/tetrahedron_triangle.cpp -lgmp -lmpfr -o intersection-reference
./intersection-reference < intersections.txt > reference.txt
```

`benchmarks/cpu/tetrahedron_side.cpp` builds with the same options. Its first
argument is the exact-coordinate input surface exported as OFF; standard input
contains a cell ID and its twelve vertex coordinates per line. It outputs
the ID and exact centroid side (`1` inside, `-1` outside, `0` boundary).
Original driver paths in the archive can be remapped to the included files.
Compact results and archive hashes are in the
[evidence index](../experiments/corner-intersections/).
