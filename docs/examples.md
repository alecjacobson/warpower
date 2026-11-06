# Reproducible full-mesher examples

The gallery runner supplies fixed inputs, runs the complete public device
pipeline, and exports an inspectable tetrahedral mesh. Surface generation or OFF
loading happens before meshing. Each captured replay includes domain and optional
sizing refits, fresh sampling, progressive fitting/refinement, quality improvement
and extraction. Host readback, independent checking and file export happen after
replay. The examples do not add a CPU geometry fallback to the mesher.

From a source checkout:

```bash
pip install -e '.[dev,bench]'
python -m examples.vtm_gallery --case torus --n 1000 --output results/local/torus
python -m examples.vtm_gallery --case graded-torus --n 1000 --output results/local/graded-torus
python -m examples.vtm_gallery --case thin-hand --n 2000 --output results/local/thin-hand
python -m examples.vtm_gallery --case cavity --n 1000 --output results/local/cavity
python -m examples.vtm_gallery --case disconnected --n 1000 --output results/local/disconnected
```

The uniform and prescribed-graded torus have completed full solves with
independent quality and topology checks. The broader gallery is still tracked
in [issue #39](https://github.com/alecjacobson/warpower/issues/39). The thin hand
currently encounters a slow exact Delaunay recovery after refinement
([issue #93](https://github.com/alecjacobson/warpower/issues/93)); a 1,000-site
attempt exceeded ten minutes without a completed mesh. The cavity and separated
components have validated input surfaces, but their full gallery solves have
not yet been validated. These commands expose those outstanding cases too.

The thin hand is a new extruded palm with five narrow fingers and sharp corners,
not the scanned hand from Alliez et al. The torus tests nonconvex topology;
the cavity and separated components exercise material orientation and extraction.
The graded torus uses a prescribed relative sizing field, ranging from 0.35 near
the leftmost side to at most 1 elsewhere. The field's min-cone extension is
computed on device. `--automatic-sizing` on an ungraded case instead selects
the mesher's pole-based local-feature-size path.

Import an already closed, oriented triangular surface without changing its
geometry:

```bash
python -m examples.vtm_gallery --case surface --input model.off --n 2000 --output results/local/model
python -m examples.vtm_gallery --case graded-surface --input model.off --n 2000 --output results/local/graded-model
```

Input positions are converted to float32, matching the mesher's API. The exported
`input.off` writes those float32 values with 17 significant decimal digits so
native CPU comparisons can receive the same coordinates. Imported assets retain
their own provenance and licensing; this runner bundles no third-party model.

Each successful run writes:

- `result.json`: settings, seed, input hash, hardware, setup/first-replay times,
  additional replay samples, CPU quota observations, topology and quality.
- `input.off` and `boundary.off`: original input and extracted surface.
- `mesh.npz`: points, material/full-hull tetrahedra, boundary faces, roles and
  radius ratios, together with the input surface.
- `mesh.vtu`: a portable VTK tetrahedral mesh with a `radius_ratio` cell field.
  Open it in a mesh viewer and use a clipping plane to inspect interior elements.

Use `--repeats 5` for five additional complete resident-device solves. Initial
setup and the first solve are reported separately. This is not a CPU speedup
measurement. `--independent-quality` adds NumPy tetrahedron solves, incidence
checks, dihedral angles and two-sided area-weighted boundary RMS against all
input triangles. That brute-force validation may be slow on large surfaces; it
is excluded from replay timing. Boundary RMS uses three quadrature samples per
triangle and is not a Hausdorff bound. Invalid solves save diagnostics and do not
export an empty mesh as a successful result.

Create a static cutaway preview from an exported mesh:

```bash
python -m examples.vtm_render results/local/torus/mesh.npz --output results/local/torus/cutaway.png
```

The preview removes whole tetrahedra according to their centroid coordinate;
it is not a geometric clipping operation on the saved mesh. Face colors show
radius ratio with directional lighting. Mesh generation and export do not
require the renderer.


## Published torus results

Both examples start from the same 1,152-vertex, 2,304-triangle torus and produce
1,000 sites, using seed 2026, 256 initial sites and four quality rounds. They
preserve one material component and a closed boundary of Euler characteristic
zero. Independent NumPy calculations found all tetrahedra positively oriented
and reproduced the device radius ratios. VTU round trips reproduce the NPZ
coordinates, connectivity and quality values exactly.

| Measurement | Uniform | Prescribed grading |
| --- | ---: | ---: |
| Material tetrahedra | 4,225 | 4,318 |
| Boundary triangles | 1,122 | 1,062 |
| Minimum radius ratio, 3r/R | 0.09286 | 0.09825 |
| Median radius ratio | 0.87442 | 0.87460 |
| Minimum / maximum dihedral angle | 3.06° / 171.81° | 5.03° / 171.82° |
| Two-sided sampled boundary RMS / box diagonal | 0.2562% | 0.3910% |
| Relative volume error | −5.09% | −7.15% |
| Complete resident solve, median of 3 additional replays | 3.637 s | 6.659 s |
| First complete replay | 3.754 s | 6.766 s |

Timings use an NVIDIA L40 and Warp 1.19.0 at runtime commit `c0b3258`.
All measured windows had zero observed CPU quota throttling. They include fresh
sampling and the complete captured pipeline. They are historical measurements,
not promises for later versions or a CPU speedup result. The native CPU baseline
currently runs faster; see [the full-mesher comparison](cpu-mesher-baseline.md).
The boundary metric is sampled RMS, not a maximum-distance guarantee. Neither
example is claimed to meet a prescribed minimum element-quality threshold.

The grading changes the allocation of sites substantially. In the regions
`x < -1` and `x > 1`, respectively, the uniform mesh has 351 / 319 sites and
median nearest-site distances 0.275 / 0.282. The graded mesh has 665 / 129 sites
and distances 0.204 / 0.404. Median tetrahedron volumes in those regions change
from 0.00412 / 0.00445 to 0.00173 / 0.01199. At the same site budget, the finer
left side comes with a coarser right side and greater total boundary-fit error.

| Uniform | Prescribed grading |
| --- | --- |
| ![Uniform torus cutaway](../results/examples/vtm/torus/cutaway.png) | ![Graded torus cutaway](../results/examples/vtm/graded-torus/cutaway.png) |

These are centroid cutaways colored by radius ratio. Download the
[uniform mesh](../results/examples/vtm/torus/mesh.vtu) or
[graded mesh](../results/examples/vtm/graded-torus/mesh.vtu) for geometric
inspection. Each result directory also contains the input/output OFF surfaces,
NPZ arrays and raw timing/quality JSON. The
[provenance manifest](../results/examples/vtm/provenance.json) records runner
revisions and artifact hashes; the
[grading/export check](../results/examples/vtm/grading-and-export-check.json)
records the regional measurements and VTU round trip.

Reproduce the measured configurations with:

```bash
python -m examples.vtm_gallery --case torus --n 1000 --repeats 3 --independent-quality --output results/local/torus
python -m examples.vtm_gallery --case graded-torus --n 1000 --repeats 3 --independent-quality --output results/local/graded-torus
```
