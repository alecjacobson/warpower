# Native Warp surface queries

The optional `SurfaceDomain(query_backend="native")` backend uses Warp's native
FP32 closest-point queries for domain queries, tetrahedron selection/extraction,
and boundary-distance metrics. The default is `query_backend="reference"`.
This work follows the [smooth-input performance plan](https://github.com/alecjacobson/warpower/issues/264).

```python
from warpower import SurfaceDomain, VariationalMeshingWorkspace
import warp as wp

domain = SurfaceDomain(vertices, triangles, query_backend="native",
                       validate_geometry=False)  # Trusted valid input.
mesher = VariationalMeshingWorkspace(domain, 8192, initial_count=256)
with wp.ScopedCapture() as capture:
    # The input surface stays fixed; upstream device kernels may update the
    # mesher's documented live controls, and downstream kernels consume output.
    mesher.run()
wp.capture_launch(capture.graph)
```

`validate_geometry` is independent of backend selection and remains `True` by
default. Disabling it skips expensive input intersection/orientation validation;
index, topology, nonfinite and degenerate-face checks remain. It does not disable
generated-mesh certification, material topology checks or transactional recovery.

## Arithmetic and scope

Signed queries use `wp.mesh_query_point` with its three oriented-ray votes.
The initial normal-based query was rejected after it misclassified the center of
a cavity: normals from equally distant opposing faces cancel. Distance-only
queries use `wp.mesh_query_point_no_sign`. Successful native queries are not
rechecked with the reference traversal. Query points and local triangle arithmetic
are FP32; physical tetrahedron centers, volumes, radius ratios and metric sums
retain FP64 and the existing wide fallback. Delaunay predicates and certification
are unchanged.

Use well-scaled, watertight, consistently oriented, non-self-intersecting surfaces
with triangles that Warp's native query accepts. Warp excludes triangles with
relative twice-area below `1e-6`; this backend does not repair or conservatively
retest those triangles. The reference domain accepts triangles down to `1e-12`
and keeps its existing behavior. FP32 queries can change boundary decisions and
optimization trajectories. Equal connectivity or bitwise coordinates are not the
accuracy contract for this explicitly selected mode.

The reference surface hierarchy remains for input validation, initial sampling
and containment/exterior recovery. The native BVH is additional setup storage.
Native selection avoids allocating the reference query cache. This backend does
not change the nearest-site quadrature accumulation; selectable accumulation
precision is tracked separately in [#266](https://github.com/alecjacobson/warpower/issues/266).

## Capture limitation in Warp 1.19

Queries and complete meshing against a fixed surface remain allocation-free in
captured loops, including conditional subgraphs. They use stable public buffers.
If surface vertices change, call `domain.refit()` before querying, as usual.
Native refit works in a top-level capture and executes entirely on device, but
Warp's internal average-edge-length scan contributes device `MEM_ALLOC` and
`MEM_FREE` nodes. Consequently native refit cannot be placed inside a conditional
subgraph. Do not omit refit after deformation.

The default reference backend retains its stronger allocation-free refit contract,
including nested captures. [#267](https://github.com/alecjacobson/warpower/issues/267)
tracks native persistent scratch support. Wrapping native refit in a conditional
child does not resolve this: scan lacks the side-allocation cache used by sort.
No native pointer patching or custom C++/CUDA is introduced.

## Reproduction

```bash
python -m examples.vtm_gallery --case surface --input armadillo.off \
  --n 8192 --initial 256 --seed 2026 --subdivisions 1 \
  --crease-subdivisions 1 --feature-angle 120 --corner-angle 45 \
  --quality-rounds 4 --repeats 3 --query-backend native \
  --no-validate-geometry --output results/local/armadillo-native
```

Use `--query-backend reference` with the same other options for the backend
ablation. The input is the pinned reduced Stanford Armadillo described in the
[gallery](examples.md). The main README's earned Paragram result is unchanged.
