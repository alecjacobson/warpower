# Scanned Armadillo example

The uniform and prescribed-graded examples mesh a closed Stanford Armadillo surface with 43,243 input
vertices and 86,482 triangles. It uses the reduced distribution in pinned libigl
tutorial-data, attributed to the Stanford Computer Graphics Laboratory. This is
an additional scanned benchmark alongside the examples inspired by Alliez et al.
Armadillo is not one of the models illustrated in that paper.

The downloader verifies the source SHA256 before centering the bounding box,
dividing coordinates by half its largest extent, and converting to float32.
Connectivity is unchanged; no holes are filled and no geometry is simplified or
repaired. Source identity, normalization and output hash are retained in
[input provenance](../results/examples/vtm/armadillo-uniform/input-provenance.json).
The [Stanford terms](https://graphics.stanford.edu/data/3Dscanrep/) permit research
use and free redistribution with attribution; commercial use requires permission.
The [data notice](../results/examples/vtm/armadillo-uniform/NOTICE.md) applies to
the model-derived arrays and figures separately from the repository code license.

From a source checkout with example dependencies:

```bash
python -m examples.fetch_armadillo --output results/local/armadillo-input
python -m examples.vtm_gallery --case surface \
  --input results/local/armadillo-input/armadillo.off \
  --n 4096 --initial 256 --seed 2026 \
  --subdivisions 1 --crease-subdivisions 1 \
  --feature-angle 120 --corner-angle 45 --quality-rounds 4 --repeats 3 \
  --output results/local/armadillo-uniform
```

The explicit 120-degree feature threshold treats this scan as smooth: the device
finds zero creases and corners. The gallery's default feature/corner thresholds
remain 45 degrees for other examples. Quadrature subdivision one already supplies
259,446 surface samples on this dense input. Source preparation and file I/O are
outside the captured meshing loop.

## Validated uniform result

| Measurement | Uniform Armadillo |
| --- | ---: |
| Sites in full hull / referenced by material cells | 4,096 / 4,090 |
| Full-hull / material tetrahedra | 24,721 / 18,517 |
| Material boundary triangles | 3,806 |
| Material components / boundary Euler characteristic | 1 / 2 |
| Minimum / first percentile / median radius ratio, 3r/R | 0.05256 / 0.40945 / 0.89222 |
| Minimum / maximum dihedral angle | 2.54° / 175.90° |
| Two-sided sampled boundary RMS / input box diagonal | 0.2417% |
| Relative volume error | −3.1419% |
| Complete resident solve, median of three warm replays | 20.834 s |
| First complete replay | 21.729 s |
| Setup, including graph construction | 2.746 s |

The L40 / Warp 1.19.0 run starts from 256 sites, performs seven growth rounds,
one extraction-recovery round and four quality rounds. Every replay includes
input-domain refit and validation, fresh sampling, progressive growth/fitting,
quality improvement and final extraction. All measured CPU quota windows are
unthrottled. Compilation/setup, transfers, independent checking and export are
separate from resident replay timing. This result establishes no CPU speedup.

Independent CGAL 6.2.1 exact predicates verify every full-hull tetrahedron's
orientation and global empty sphere, face pairing and hull support. The resulting
material boundary has no self-intersections; independent incidence checks and
single-cycle vertex links establish its manifold topology. NumPy independently
reproduces cell quality, dihedrals and volume. The VTU arrays round-trip exactly.
The earlier exact input self-intersection result is reused only after verifying
that the normalized input OFF bytes are identical. See the
[independent checks](../results/examples/vtm/armadillo-uniform/independent-checks.json)
and [quality report](../results/examples/vtm/armadillo-uniform/independent-quality.json).

| Material surface | Interior cutaway |
| --- | --- |
| ![Armadillo material surface](../results/examples/vtm/armadillo-uniform/surface.png) | ![Armadillo tetrahedral cutaway](../results/examples/vtm/armadillo-uniform/cutaway.png) |

Colors show the incident tetrahedron's radius ratio. The cutaway removes whole
cells by centroid; it does not modify the exported mesh. Download the
[VTU](../results/examples/vtm/armadillo-uniform/mesh.vtu),
[NPZ arrays](../results/examples/vtm/armadillo-uniform/mesh.npz), or
[raw timing record](../results/examples/vtm/armadillo-uniform/result.json).
The [manifest](../results/examples/vtm/armadillo-uniform/provenance.json) pins
runtime `6387888`, example/oracle `7765cb8`, renderer revision and artifact hashes.

The preview supports explicit camera orientation and dataset credits:

```bash
python -m examples.vtm_render results/local/armadillo-uniform/mesh.npz \
  --axis z --fraction 0.6 --up-axis y --azimuth -20 --elevation 15 \
  --title 'Stanford Armadillo · tetrahedral cutaway' \
  --credit 'Stanford Computer Graphics Laboratory · libigl tutorial-data distribution' \
  --output results/local/armadillo-uniform/cutaway.png
```

## Independent distances on dense surfaces

The optional native AABB distance oracle makes offline boundary-fit validation
practical for the 86,482-triangle input. Build it as described in the
[native reference instructions](../benchmarks/cpu/README.md), then run:

```bash
python -m benchmarks.mesh_quality results/local/armadillo-uniform/mesh.npz \
  --distance-binary results/local/surface-distance \
  --output results/local/armadillo-uniform/independent-quality.json
```

This oracle is used after GPU timing. NumPy still computes cell quality and
incidence; only triangle-distance queries use the independently built CGAL tree.
The distance constructions use float64. Both directions use three quadrature
points per surface triangle, so the reported RMS is a sampled metric rather than
a Hausdorff certificate. The
[oracle audit](../results/examples/vtm/armadillo-uniform/distance-oracle-audit.json)
checks 918 translated/scaled single- and multiple-triangle queries against known
projections and the existing brute-force reference, five rejected-input cases,
empty queries and an entire two-sided quality report. Binary/source hashes and
CGAL version are recorded with each report.

The uniform artifacts above preserve their original runtime `6387888`. The
new recovery runtime `058fa60` reproduces that mesh bitwise, with identical
iteration counts and a fresh warm median of 20.838 s. The full suite has 575
passing tests. No native oracle is called by the device meshing pipeline.

## Preserved graded failure

At the preceding runtime, changing only `--case` to `graded-surface` with these
budgets failed material extraction. The existing min-cone sizing field ranges from relative
size 0.35 on the left to at most 1 elsewhere. Input validation succeeds, but the
final material selection has two edges incident to four boundary triangles.
That run returned error 64, valid 0 and zero output counts.
Its failed first replay takes 7.888 s; this is not a valid-mesh performance result.

The saved internal candidate has 4,096 sites, 24,677 independently certified
full-hull tetrahedra and 18,769 selected cells. Its two offending edges are
`(234, 1343)` and `(1296, 2965)`. Containment recovery adds no cells. The
[failed candidate and diagnostics](../results/validation/vtm/graded-armadillo-failure/NOTICE.md)
are retained separately from the valid uniform example. See the
[edge analysis](../results/validation/vtm/graded-armadillo-failure/failure-analysis.json),
[full-hull check](../results/validation/vtm/graded-armadillo-failure/full-hull-check.json)
and [failure report](../results/validation/vtm/graded-armadillo-failure/result.json).

## Validated graded result

[Exterior-band recovery](exterior-band-recovery.md) resolves this failure after
existing fitting recovery is exhausted. It removes exterior-center tolerance
cells at bad edges within a bounded transaction, preserves referenced vertices
and prescribed feature cells, and rolls back if topology or volume guards fail.
This is a boundary approximation; it does not prove removed cells wholly exterior.
The same recovered material selection is used inside the quality-stage guards.

Reproduce with the command above, changing `--case surface` to
`--case graded-surface` and the output directory to `armadillo-graded`.
The prescribed field is a clipped min-cone centered at the left face center of
the input box, with minimum 0.35, maximum 1 and slope `0.65 / box_width`.

| Measurement | Graded Armadillo |
| --- | ---: |
| Sites in full hull / referenced by material cells | 4,096 / 4,092 |
| Full-hull / material tetrahedra | 24,667 / 18,746 |
| Material boundary triangles | 3,618 |
| Material components / boundary Euler characteristic | 1 / 2 |
| Minimum / first percentile / median radius ratio, 3r/R | 0.07408 / 0.45791 / 0.89740 |
| Minimum / maximum dihedral angle | 3.61° / 173.85° |
| Two-sided sampled boundary RMS / input box diagonal | 0.2752% |
| Relative volume error | −3.6972% |
| Complete resident solve, median of three warm replays | 20.882 s |
| First complete replay | 21.792 s |
| Setup, including graph construction | 2.827 s |

This L40 / Warp 1.19.0 result uses runtime `058fa60`, the same input and budgets
as the saved failure, seven growth rounds, three recovery rounds and four quality
rounds. Every replay performs a fresh complete solve. Setup, transfer, export and
independent validation remain outside resident timing; all measured quota windows
are unthrottled. The last extraction removes two cells in one accepted pass.

Independent CGAL checks cover 100,937,364 site/sphere queries, 292 full-hull faces,
1,196,032 supporting-plane queries, face pairing and positive orientation. The
material boundary has zero self-intersection pairs and single-cycle vertex links;
NumPy independently verifies quality, volume and incidence, and the VTU round trip
is exact. Input self-intersection validation is reused only for identical OFF
bytes. See [independent checks](../results/examples/vtm/armadillo-graded/independent-checks.json),
[quality](../results/examples/vtm/armadillo-graded/independent-quality.json), and
[provenance](../results/examples/vtm/armadillo-graded/provenance.json).

The grading reallocates sites at the same total budget. In the leftmost and
rightmost thirds of the input's x extent:

| Measurement | Uniform left / right | Graded left / right |
| --- | ---: | ---: |
| Sites | 690 / 972 | 1,033 / 554 |
| Median nearest-site distance | 0.05199 / 0.05125 | 0.04427 / 0.06345 |
| Median material tetrahedron volume | 2.743e−5 / 2.668e−5 | 1.699e−5 / 4.647e−5 |

The finer left region comes with a coarser right region and slightly greater
whole-surface fit error. [Raw regional measurements](../results/examples/vtm/armadillo-graded/grading-comparison.json)
use the same bounding-box regions for both meshes.

| Graded material surface | Graded interior cutaway |
| --- | --- |
| ![Graded Armadillo surface](../results/examples/vtm/armadillo-graded/surface.png) | ![Graded Armadillo tetrahedral cutaway](../results/examples/vtm/armadillo-graded/cutaway.png) |

Download the [graded VTU](../results/examples/vtm/armadillo-graded/mesh.vtu),
[NPZ arrays](../results/examples/vtm/armadillo-graded/mesh.npz), and
[raw timing](../results/examples/vtm/armadillo-graded/result.json).
The [Stanford data notice](../results/examples/vtm/armadillo-graded/NOTICE.md)
applies to these model-derived artifacts.

This completes the uniform/graded scanned-example scope in
[issue #140](https://github.com/alecjacobson/warpower/issues/140) and the specific
recovery in [#142](https://github.com/alecjacobson/warpower/issues/142).
The broader examples, difficult-boundary cases and quality-matched CPU
outperformance goals remain open. No CPU speedup is established by these runs.
