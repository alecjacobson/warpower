# Captured domain extraction

`ExtractionWorkspace` selects a domain subcomplex from an existing tetrahedral
mesh. It is independently callable, preserves the input connectivity, and
composes with `DiagramWorkspace` or the internal VTM core in a captured loop.

```python
import warp as wp
from warpower import DiagramWorkspace, ExtractionWorkspace, SurfaceDomain

domain = SurfaceDomain(vertices, triangles)
mesh = DiagramWorkspace(sites, adjacency=False, return_tetrahedra=True)
extract = ExtractionWorkspace(
    domain, mesh.points, mesh.tetrahedra, mesh.tetrahedra_count,
    source_error=mesh.error, tolerance=0.4,
)
with wp.ScopedCapture() as capture:
    mesh.run()
    extract.run()
    # A downstream kernel reads extract.valid, tetrahedra and tetrahedra_count.
wp.capture_launch(capture.graph)
```

Setup allocates, compiles, captures, and synchronizes. `run(stream=None)` uses
stable buffers and device counts, without host geometry decisions, transfers,
allocations, or callbacks. Upstream processes can change positions, connectivity,
and the input count before replay. Call `domain.refit()` if its vertex positions
change, and establish stream dependencies with other producers/consumers.
Contiguous CUDA Warp/PyTorch inputs share storage; NumPy arrays upload at setup.

## Selection rule

The default follows Section 3.7 of
[Alliez et al.](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf):
retain a tetrahedron if its circumcenter is inside the domain, or its outside
distance to the boundary divided by its circumradius is **strictly less than
0.4**. Set `tolerance=0` to retain only centers in the closed domain. The supported
range is `[0, 1)`. Calculations use float64 on float32 input positions.

This is boundary approximation. Retained tetrahedra may extend beyond the input
surface, and an outside circumcenter can be retained intentionally. The rule
does not guarantee topology or preserve input triangles exactly. Sampling,
boundary fitting, and adequate site resolution remain necessary.

Optional [containment recovery](containment-recovery.md) can add proven wholly
material cells around a bad boundary edge. Enable it with
`containment_recovery=True`; the independent extractor defaults to `False`.
The final selection includes these additions, so it need not equal the
circumcenter predicate alone. Remaining topology failures are still explicit.

Optional [exterior-band recovery](exterior-band-recovery.md) can remove
exterior-center tolerance cells around bad boundary edges, with vertex, feature,
volume and rollback guards. Enable it with `exterior_recovery=True`; its
independent default is also `False`. This is boundary approximation, not proof
that a removed cell is wholly outside material.

Inputs must be a nonoverlapping, positively oriented tetrahedral complex. For
the paper's interpretation, use the complete Delaunay triangulation. Extraction
checks index validity, finite referenced positions, positive determinants, and
finite circumcenter calculations. Cancellation-prone cells use a
[wide integer-geometry fallback](extraction-geometry.md) for physical determinant
and circumcenter estimates; extraction centers remain float64. It does **not**
independently certify empty spheres or test arbitrary tetrahedron intersections.
`DelaunayWorkspace` provides separate original-coordinate full-hull certification.
Weighted regular tetrahedra can be
passed as connectivity, but this extractor always uses **ordinary geometric
circumcenters**, not power centers.

## Output and topology diagnostics

`tetrahedra[:tetrahedra_count[0]]` is the selected mesh, preserving input order.
`source_indices` maps that prefix to input tetrahedra. Per-input `selected`,
`circumcenters`, `circumradii`, and `signed_distance` expose the classification;
only the active input prefix has meaningful geometric statistics. Unused output
storage is unspecified. The input count may be zero and may change on device.

Outward `boundary_faces[:boundary_count[0]]` and `boundary_owners` identify exposed
triangles and their owning output tetrahedra. Face output order is unspecified.
Union operations produce device `component_labels` for face-connected tetrahedra,
`shell_labels` for edge-connected boundary triangles, and their respective
`component_count`/`shell_count`. Labels are root indices, not contiguous IDs.

`shell_euler` and `shell_volume` contain values only at shell-root indices; other
entries are zero. A closed orientable shell has Euler characteristic `2 - 2g`.
Outward material shells have positive signed volume and cavity shells negative
volume. `cavity_count` counts the negative shells when topology checks pass. These
diagnostics describe the **output**; they do not establish equality with the
input domain's topology. A torus has one shell with Euler characteristic zero;
an enclosed spherical-type cavity adds a negative shell with characteristic two.

Device `diagnostics[0]` is a bit mask:

| Bit | Detected topology problem |
|---|---|
| 1 | Face incidence greater than two, or equally oriented paired faces |
| 2 | Boundary edge incidence other than two, or equally oriented paired edges |
| 4 | Boundary vertex link is not one cycle, including point contacts |

These checks retain the selected subcomplex for inspection. `valid[0]` equals one
only when both `error` and `diagnostics` are zero. It means the documented checks
passed, **not** a general geometric or mesh-quality certificate. An empty output
also passes these checks; consumers must inspect the device count if they require
a nonempty mesh. On a fatal `error`, both output counts are zero.

| Error bit | Fatal input or query failure |
|---|---|
| 1 | Invalid domain |
| 2 | Invalid input count or nonzero upstream `source_error` |
| 4 | Invalid vertex index or nonfinite referenced position |
| 8 | Nonpositive determinant or invalid circumcenter/radius |
| 16 | Failed nearest-boundary query or unresolved inside/outside classification |

Corrected input can recover on the next replay. Statistics are invalid on fatal
error. All topology tables are preallocated for the input tetrahedron capacity;
memory and launch work scale with that capacity, even if the active prefix is
small. Extraction and its diagnostics are opt-in: standalone diagram/Delaunay
calls do not allocate or execute this layer.

## Quality and approximation measurements

`radius_ratio` stores `3 * inradius / circumradius` for each output tetrahedron,
and `minimum_radius_ratio` reduces the active prefix. For empty output the latter
retains its initialization value of one; it is not a quality result.
`volume`, `domain_volume`, `volume_difference`, and `boundary_area` measure the
selected complex and oriented input triangle domain. Summed shell volumes
provide an independent decomposition of the selected volume.

`surface_rms_distance` and `surface_max_sample_distance` use three degree-two
quadrature nodes per extracted boundary triangle, measuring distance to the
input triangles. RMS weights the squared distance by surface area. These are
**directed sampled measurements**, not symmetric distances or Hausdorff bounds;
thin missed features can escape them. The quadrature rule is degree two, while
distance to a piecewise surface generally is not a polynomial.

Run the torus extraction benchmark with independent CPU classification and
topology checks:

```bash
python -m benchmarks.extraction --reference --output results/local/extraction.json
# Include the preceding optimization in the resident producer/consumer loop:
python -m benchmarks.extraction --combined --reference --output results/local/combined.json
```

Graded sizing, initialization/refinement, and worst-element improvement remain
part of the [full mesher development goal](vtm.md).
