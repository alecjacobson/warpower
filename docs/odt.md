# Fixed-boundary ODT optimizer

`ODTWorkspace` is the first optimization component of the
[VTM implementation](vtm.md). It optimizes uniform tetrahedral interpolation
energy inside the initial convex hull, holding the hull vertices fixed. It does
not fit a supplied surface, extract a nonconvex domain, grade element sizes,
or guarantee good worst-element quality. The [public full mesher](mesher.md)
composes boundary fitting, graded sizing, extraction and quality stages separately.

```python
import warp as wp
import warpower

# Contiguous CUDA wp.vec3 sites, updated in place.
optimizer = warpower.ODTWorkspace(points, max_iterations=20)
with wp.ScopedCapture() as capture:
    # Optional upstream kernels write points here.
    optimizer.run()
    # Downstream kernels consume optimizer.points / tetrahedra / tetrahedra_count.
wp.capture_launch(capture.graph)
```

As with `DiagramWorkspace`, allocation, compilation, and capture occur during
setup. All optimization decisions, tetrahedralization, reductions, rejection,
rollback, and convergence checks run on the device. `run()` also works inside an
enclosing `wp.capture_while` loop. Output buffers have stable pointers and device
counts. Consumers must guard on `error[0] == 0` and use only the published prefix.
Different concurrent executions require separate workspaces. Use Warp capture
when composing with PyTorch on a shared stream.

## Numerical method

For a positively oriented tetrahedron of volume V, the uniform ODT contribution
is `V / 20 * sum(squared lengths of its six edges)`. The implementation evaluates
this expression in float64. It accumulates each incident tetrahedron's volume
and `V * (circumcenter - site)` to form interior relocation directions. The latter
uses cofactors without first dividing by a nearly zero determinant. Sites remain
float32, matching the standalone geometry input contract.

Uniform and graded accumulation retain the usual binary64 determinant when its
sign is certified by a conservative error bound. Nearly cancelling cases use
an exact integer orientation of the original sites and its physical determinant
estimate. True inverted/zero cells still fail. See [robust volumes](stable-volumes.md)
for the arithmetic argument, real failing cells, and validation.

The initial hull vertices are detected from the validated face incidence index.
The optimizer caps each proposed displacement by a fraction of the site's shortest
incident edge, damps it, and independently limits it against the initial hull
planes. A coordinate-scale margin accounts for rounding back to float32. This is
a practical GPU update policy, not a claim of exact sequential minimization.

Each trial reconstructs Delaunay connectivity and checks the actual new energy
and hull. Requiring all *old* tetrahedra to stay positively oriented would prevent
legitimate connectivity changes and can freeze optimization. A rejected trial
halves the damping on device. Exhausting the backtracking budget restores both
the old positions and their validated connectivity. This handles a rejected
numerical reconstruction without allowing stale/partial output to escape.

The underlying full-hull builder still has the documented degeneracy limitations.
Initial failure propagates through `error`; a failure on a proposed step may be
recovered through backtracking. Robust construction is separately tracked in
[#32](https://github.com/alecjacobson/warpower/issues/32).

## Parameters and results

| Parameter | Meaning |
|---|---|
| `max_iterations=20` | Maximum accepted steps per run; zero only constructs/evaluates |
| `damping=0.5` | Initial multiplier, halved after rejection |
| `step_limit=0.5` | Maximum undamped displacement / shortest incident edge |
| `tolerance=1e-4` | Stop when maximum accepted displacement / shortest edge is this small |
| `max_backtracks=12` | Maximum halvings after the first rejected attempt |
| `fixed=None` | Optional contiguous CUDA int32 mask pinning additional sites |
| `tetrahedra_capacity`, `chunk_size` | Forwarded to the full-hull workspace |
| `tetrahedra_cell_capacity=1024` | Larger clipping recovery for difficult intermediate cells; set 512 for the smaller reserve |

Every `run()` uses the current sites as its initial state, rediscovers their hull,
and resets iteration counters and history. An uploaded NumPy input is copied;
a CUDA Warp input is modified in place. A user-supplied fixed mask remains bound
by pointer and may be updated in place before a run.

| Device output | Meaning |
|---|---|
| `points`, `tetrahedra`, `tetrahedra_count` | Current accepted full-hull mesh |
| `fixed` | Effective mask including the initial hull |
| `energy`, `volume`, `minimum_radius_ratio` | Final accepted geometry statistics |
| `iterations` | Number of accepted updates |
| `energy_history` | Initial energy plus `iterations` accepted energies; remaining entries are NaN |
| `rejected_steps` | Number of rejected candidate steps |
| `termination` | 0: iteration limit; 1: displacement threshold; 2: exhausted backtracking |
| `error` | Initial geometry error bits; 256: invalid geometric arithmetic; 1024: hull-plane capacity |
| `last_geometry_error` | Most recent rejected Delaunay construction error, or zero |

Termination 2 returns the last accepted valid mesh, not a successful convergence
claim. Displacement convergence is also not a proof of global optimality.
`minimum_radius_ratio` is `3 * inradius / circumradius`; it can worsen even while
ODT energy decreases. Fixed poorly shaped hull facets and sites approaching the
boundary need the later surface-fitting and quality-improvement stages.

The reductions use floating-point atomics; summation order and thus trajectories
are not promised to be bitwise identical across runs/devices. Energy acceptance
allows relative reduction roundoff of `1e-12`. Raw graph buffers beyond their
valid prefixes are unspecified.

Run the numerical and capture tests with `python -m pytest tests/test_odt.py`.
`python -m benchmarks.odt --output results/local/odt.json` measures this optimizer
only. It is not the full mesher or a CPU speedup comparison.
