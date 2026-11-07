# Optional local quality backtracking

`VariationalMeshingWorkspace(..., local_quality_retries=12)` can keep a
selective proposal moving when a few tetrahedra reject the shared step. The
default is zero: it retains the original proposal kernels, storage and graph.
This is an optional optimization heuristic, with shape-dependent quality/time
tradeoffs. It does not change the fitting objective or add surface projection.

```python
mesher = warpower.VariationalMeshingWorkspace(
    domain, 4096, local_quality_retries=12,
)
with wp.ScopedCapture() as capture:
    domain.refit()
    mesher.run()
```

The [gallery](examples.md) accepts the same option:

```bash
python -m examples.vtm_gallery --case counterbored-plate --n 1000 \
  --local-quality-retries 12 --repeats 3 --output results/local/plate-local
```

## Transaction and fallback

Only a new rejection by the selective minimum-quality guard is eligible.
The implementation compares the rejection counter before and after the current
trial; a sticky error from an earlier trial cannot authorize a local retry.
Geometry, classification and objective failures use ordinary global backtracking.

The current retained tetrahedra whose radius ratios fall below the saved minimum
identify candidate vertices. Moved, selected vertices of those cells are blocked
for the next trial, which restores their original coordinates exactly. Other
vertices keep the current shared step. Blocking can accumulate across local
retries within one optimizer step. Original corner anchors, selective masks,
active-prefix handling and movement caps still apply.

Every trial reconstructs and certifies the complete current Delaunay mesh and
reruns the ordinary boundary/energy and selective quality checks. There is no
partial connectivity publication or weaker acceptance threshold. This is not a
proof that blocking those vertices removes the conflict: a newly appearing cell
can be caused by moving another site. No newly blocked vertices, an unsupported
failure, or an exhausted local budget restores the complete original direction
and resumes global backtracking. Exhausting the global budget restores the
starting points, connectivity and statistics through the existing transaction.

`local_quality_retries` is a setup integer in `[0, 2**31)`. It bounds additional
trials per optimizer step, beyond the core's `max_backtracks` budget. There are
at most `1 + max_backtracks + local_quality_retries` attempted reconstructions
per step. A positive setting requires quality rounds in the public mesher, or
`selective_threshold` in the internal core. Quality-stage and iteration budgets
retain their existing meanings.

## Device state and limitations

The option adds an int32 block mask of one entry per reserved site and six
int32 scalar entries, or `4 * capacity + 24` bytes of explicit array storage.
Graph/runtime storage is additional. All masks, reductions, eligibility checks,
retry decisions and resets remain on device. Fresh core invocations and optimizer
steps reset the appropriate state; ordinary fallback clears the block mask.
No host allocation, readback, synchronization or graph recapture is needed during
replay. Independent diagram and Delaunay constructors do not allocate this state
or import the helper.

`core.local_backtracking.statistics` records local retries and vertex-blocking
events during the most recent core invocation. `core.rejected_steps` and the
selective rejection counter include rejected local trials. These child counters
are not accumulated totals for the complete mesher scheduler.

The selective guard preserves its starting material minimum for accepted
selective steps. This does not guarantee the same final mesh, a better minimum,
or less work than a solve with the option disabled. Other quality stages retain
their existing acceptance rules. An accepted zero-movement proposal can terminate
the core even though blocked directions remain. As with the original movement
test, that termination does not establish optimality.

The motivating [midpoint/hand investigation](representable-boundary-probe.md)
and [issue #202](https://github.com/alecjacobson/warpower/issues/202) track the
quality and performance evaluation. The midpoint correction itself remains
unadopted. This option uses ordinary rounding.
