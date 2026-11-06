# Device-resident CUDA graph loops

`DiagramWorkspace` preallocates storage and captures the construction once.
Every `run()` then enqueues GPU work without reading counts, bounds, or status
back to the CPU. It supports bounded Voronoi/power adjacency and optional full-hull
Delaunay/regular tetrahedra. The existing convenience functions retain their
exact-size results and host orchestration; use the workspace for captured loops.

```python
import warp as wp
import warpower

# points: contiguous CUDA wp.vec3 array; weights: optional float32 CUDA array.
workspace = warpower.DiagramWorkspace(
    points, weights,
    return_tetrahedra=True,
)

with wp.ScopedCapture() as capture:
    # Launch an upstream kernel that writes points/weights in place here.
    workspace.run()
    # Launch downstream kernels reading workspace outputs here.

for _ in range(100):
    wp.capture_launch(capture.graph)
```

`workspace.run()` also works outside an enclosing capture: it asynchronously
launches its internally captured graph. Python is used for setup and graph
submission, not for convergence decisions during execution. Setup may allocate,
compile, upload NumPy inputs, and prepare native sorting scratch. Replays keep
geometry, bounds, counters, validation results, and decisions on the device.
The constructor finishes initialization on its setup stream before returning,
so the prepared workspace can then be used on another properly ordered stream.
CUDA conditional graphs require Warp's CUDA toolkit and the NVIDIA driver to
support CUDA 12.4 or newer.

`run()` can also be nested inside a `wp.capture_while` body alongside producer
and consumer kernels. Updating that loop's condition on the GPU allows multiple
construction/consumption iterations in a single graph launch, with no host
intervention between iterations.

## Outputs and capacities

All outputs are CUDA Warp arrays with stable addresses for the workspace lifetime.
Their physical shapes are capacities, not the current number of valid entries.

| Field | Meaning |
|---|---|
| `adjacency` | int32 capacity buffer for CSR neighbors |
| `offsets` | int32 `(N+1,)` CSR offsets; use these for row traversal |
| `adjacency_count` | int32 `(1,)`, valid prefix length of `adjacency` |
| `status` | int32 `(N,)` bounded-cell statuses when adjacency is requested |
| `tetrahedra` | int32 `(capacity,4)`, positively oriented input-point indices |
| `tetrahedra_count` | int32 `(1,)`, valid prefix length of `tetrahedra` |
| `tetrahedra_expansions` | int32 `(1,)`, expansions in the current replay; present when tetrahedra are requested |
| `error` | int32 `(1,)` bitmask; zero means the requested outputs are valid |

For a tetrahedron consumer, launch over capacity and read the count on the GPU:

```python
@wp.kernel
def consume_tets(tets: wp.array2d[int], count: wp.array[int], error: wp.array[int]):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        a = tets[i, 0]
        # Use this tetrahedron; do not read beyond count[0].
```

Default capacities are `48*N` adjacency entries and `max(16, 8*N)` tetrahedra.
They are practical reserves, not theoretical bounds on arbitrary 3D input.
Set `adjacency_capacity=` and `tetrahedra_capacity=` to reserve more before
capture. The tetrahedron capacity also bounds the intermediate candidate list.
The current workspace conservatively requires `512*N < 2**31` (at most
4,194,303 sites) and checks output capacities against int32 indexing limits.
There is no automatic allocation inside a replay. If a reserve is exhausted,
construct a larger workspace outside capture and recapture the enclosing graph.
`adjacency=False, return_tetrahedra=True` skips the bounded adjacency computation.
The `status` field is then unused, initialized to -1.

As in the eager API, the bounded adjacency can omit boundary edges of the full
tetrahedralization. Weighted sites with globally empty power cells are omitted
from the regular mesh. The mesh has the same floating-point and degeneracy
[limitations](api.md#optional-full-tetrahedral-mesh) as the eager optional output.

## Failure handling on the device

Consumers must check `error[0]` or the published counts. On any error, both output
counts and all CSR offsets are zeroed before downstream work runs, so a previous
successful replay cannot appear as current output. Buffer entries outside the
valid prefixes are unspecified and may contain old data.

| Bit | Meaning |
|---:|---|
| 1 | Nonfinite point or weight input |
| 2 | Cell construction exhausted its capacity/numerical recovery |
| 4 | Adjacency capacity exceeded or CSR count overflow |
| 8 | Tetrahedron candidate capacity exceeded |
| 16 | Detected cospherical/copower or near-coplanar ambiguity |
| 32 | Power-sphere or mesh topology validation failed |
| 64 | Full-hull domain expansion limit exhausted |

Degenerate inputs can fail cell construction before the more specific ambiguity
test runs. Error flags reset on every replay, so corrected input can succeed
using the same graph. `max_expansions` defaults to 12.

## Lifetimes, streams, and changing inputs

Keep the workspace and its bound input arrays alive as long as the graph uses
them. Update input contents in place; replacing a Python array variable does not
rebind a captured pointer. Point count, dtype, capacities, and enabled outputs are
fixed during setup. Different topology and output counts are supported on every
replay. Noncontiguous CUDA inputs are rejected to preserve pointer identity.
NumPy inputs are uploaded during setup; later changes to the NumPy source do not
update the device copy.

By default `run()` uses the current Warp stream. Pass `stream=` or use
`wp.ScopedStream` to select another. For CUDA tensors, wrap the PyTorch stream with
`wp.stream_from_torch()` and put both frameworks' producer and consumer operations
on that shared stream. Use **Warp `ScopedCapture`** for the enclosing capture so
Warp can insert its conditional graph nodes; a capture started solely through
`torch.cuda.graph` is not accepted. Wrap output arrays with `wp.to_torch()` once
outside capture when a PyTorch consumer needs tensor views.

Separate concurrent executions need separate workspaces. Reusing one workspace
on different streams requires ordering them with events; its mutable scratch and
outputs cannot be used concurrently.

## Device control flow

The graph rebuilds the Morton hierarchy for the current points and weights.
Conditional nodes run failed-cell retries and adjacency repairs only as needed.
A preallocated Warp tile scan forms CSR offsets without allocating scan scratch
inside a conditional branch. The optional dual uses device-controlled chunk and
domain-expansion loops, with the same empty-power-sphere and hull-boundary checks
as the eager implementation.

The last successful domain scale is retained on the GPU as a starting point for
the next tetrahedralization. Every new mesh is still reconstructed and validated;
the domain expands again if necessary. This avoids repeating the full boundary
search for every frame. An unsuccessful mesh resets that starting scale.

The [graph tests](../tests/test_graph.py) exercise producer/construction/consumer
capture, changing topology, capacity exhaustion, failure recovery, weighted
hidden sites, and independent tetrahedralization references. The
[benchmark script](../benchmarks/graphs.py) compares warmed graph replay against
the eager API and checks output equivalence.

The [validation report](graph-validation.md) records the measured speedups,
profiler inspection, memory tradeoff, and reproduction commands.
