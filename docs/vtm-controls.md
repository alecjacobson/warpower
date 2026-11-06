# Device controls for the VTM core

The private `warpower._vtm.VTMWorkspace` accepts optional device scalars for
its iteration budget and global/selective mode. A scheduler can reuse one
optimizer, mesh, and history allocation across refinement and polishing phases.
Standalone diagram and Delaunay APIs remain independent.

```python
import warp as wp
from warpower._vtm import VTMWorkspace

budget = wp.full(1, 5, dtype=int, device=points.device)
selective = wp.zeros(1, dtype=int, device=points.device)
core = VTMWorkspace(
    domain, points, max_iterations=50,
    iteration_limit=budget,
    selective_threshold=0.3, selective_enabled=selective,
)
with wp.ScopedCapture() as capture:
    # A producer may set budget/selective and update points here.
    core.run()
    # Consume the full-hull tetrahedra and count on device.
```

Each control must be a contiguous int32 `[1]` array on the core device. The
caller owns it and may update it between calls, including with upstream device
kernels. As with the other stateful workspaces, inputs and controls must remain
stable while a call executes.

`iteration_limit[0]` is a per-call upper bound in `[0, max_iterations]`.
`max_iterations` remains the setup bound and allocates the history capacity.
Zero performs the normal geometry/statistics refresh and corner anchoring,
without optimization steps. Convergence, failed backtracking, or no selected
movable vertices can stop a valid call before its budget is exhausted. Each
call restarts its iteration count and history; this scalar is not a cumulative
budget across calls.

`selective_enabled[0]` is 0 for global updates and 1 for
[selective updates](selective.md). It requires a `selective_threshold` at setup.
Global mode fills the update mask and skips retained selection and the extra
retained-quality guard. Both modes retain ordinary boundary/ODT acceptance,
certified reconstruction, and rollback. Selection scratch is meaningful only
after a selective step; a zero-step refresh may leave scratch from a prior call.

Either control can be omitted independently. Without `iteration_limit`, the
setup iteration limit applies. Without `selective_enabled`, the configured
global or selective behavior is fixed. Omitting both preserves the original
device work, kernels, and allocation choices.

An invalid runtime budget or mode sets core error bit **32**, stops the
optimization loop, and publishes zero tetrahedra. Validation takes place after
the ordinary initial geometry refresh, so the source points can already have
their normal exact corner anchors. A later valid replay reconstructs current
inputs and recovers normally. Consumers must gate on the core error/count;
scratch arrays are not published valid output on error.

## Validation

Tests compare controlled and fixed-mode trajectories for budgets 0, 1, and 3;
they check positions, connectivity, accepted/rejected counts, termination, and
energy histories. Further cases cover graded active prefixes, protected
corners, each control independently, invalid values and setup arrays, and a
captured producer/core/consumer sequence that changes both controls and recovers
from failures.

The [source audit](../results/validation/vtm/controls-isolation.json) finds 39
other runtime modules unchanged from `8e38177`. All eight preexisting VTM and
all eight selective kernel/function ASTs are unchanged. Only optional Python
recording paths and a separate control kernel are added.

The default 1,000-site torus/20-iteration benchmark retains 5,902 tetrahedra,
20 accepted steps, zero rejections, and matching roles and energy history
(relative tolerance `2e-14`). Raw before/after measurements are in
[baseline](../results/validation/vtm/controls-default-before.json) and
[current](../results/validation/vtm/controls-default-after.json). This measures
default-path preservation, not full-mesher CPU performance.
Median times are 521.51 ms before and 520.45 ms after (−0.20%, within measurement
variation), with no CPU quota throttling in either run.

```sh
python -m benchmarks.control_loop --dot results/local/controls.dot \
  --output results/local/controls.json
```

The audit driver uses six device-produced phases: zero-step refresh, two global
steps, invalid budget, up to two selective steps, invalid mode, and one global
step. Device consumers record each phase's error, iteration/tetrahedron counts,
and volume. Host readback occurs after all replays finish.

All [10 focused tests](../results/validation/vtm/controls-tests.txt) and
[340 full-suite tests](../results/validation/vtm/controls-full-tests.txt) pass.
The captured changing-control case passes
[full-memory and allocation-lifetime checks](../results/validation/vtm/controls-lifetime.txt),
and graded active prefixes pass [pooled memory checks](../results/validation/vtm/controls-pooled-memcheck.txt),
both with zero sanitizer errors using the documented [diagnostic settings](cuda-validation.md).
The [native audit](../results/validation/vtm/controls-device-audit.json) records
2,117 kernel activities across six replays. All 59 graph copy nodes are
device-to-device; there are no host, allocation/free, or CPU-transfer nodes.
Nsight reports no copy activities for this graph, so copy direction is verified
from the complete DOT graph rather than inferred from an empty activity list.
