# Initial site budgets and mandatory corners

The complete mesher now treats `initial_count` as a minimum. Before sampling,
each replay resolves `min(target, max(minimum, corner_count + 4))` on device.
When the minimum is automatic, it follows the current target up to 256. The four
additional samples leave volume sites after anchoring when room permits.
An explicit minimum still must fit the target; a target that cannot hold the
mandatory corners reports input error 1 and zero published counts.

The resolved device scalar is `effective_initial_count`. The gallery reports it
alongside `minimum_initial_sites`, and explains a target/corner capacity failure.
No corner is silently discarded. Initializing with sufficient sites is not a
promise of valid final material topology.

The mesher owns a private sampler with a device-selected active prefix. It reserves
storage for up to `min(capacity, max(minimum, input_vertex_count + 4))` samples at
setup, validates the requested prefix and publishes its count on device. Inactive
slots do not participate in duplicate detection. Sampling the same prefix gives
the same points and rejection counts as the public fixed-count sampler. The
independently callable `InitialSamplingWorkspace` and its kernels are unchanged.

## Controls and regression checks

On L40 / Warp 1.19.0, median of five additional complete fresh captured solves:

| Reference backend configuration | Before | After |
|---|---:|---:|
| Uniform Armadillo 8K, feature angle 120, one subdivision | 3473.43 ms | 3451.56 ms |
| Graded Armadillo 8K, same surface configuration | 3676.36 ms | 3679.73 ms |
| Torus 1K | 489.39 ms | 487.40 ms |
| Counterbored plate 1K | 867.93 ms | 866.50 ms |

All four pairs have bitwise-identical sites and equal roles, full/material
connectivity and oriented boundary faces. The candidate meshes pass independent
exact full-hull/Delaunay, material-incidence, boundary-link, self-intersection and
VTU round-trip checks. These are regression controls, not a CPU-speedup claim.
The public power/regular-diagram performance matrix passes its paired 2% gate.
Standalone power, Delaunay and fixed-count sampling sources are byte-identical.

The full ordinary suite passes 1,417 tests. An initial run omitted three historical
fixtures from the sparse checkout, produced 13 missing-file failures and was
interrupted. The fixtures were restored with matching Git/blob and SHA-256 hashes
before the successful rerun. Both runs are retained. All six new cases also pass
isolated padded and allocation-lifetime memory checks (12 runs, zero reported
errors), including changed target and corner counts inside captured fresh solves. The first plate timing command
used an invalid example name; the corrected `counterbored-plate` pair is reported
above, with the failed invocations preserved.

## Default Armadillo remains unresolved

The default 1K Armadillo detects 972 mandatory corners at the 45-degree feature
threshold. Initialization now grows from 256 to 976 sites and reaches 1K after
86 fitting steps. The final material boundary still fails validation, even after
three recovery rounds. Zero mesh counts are published, as required for failure.

Larger targets do not by themselves solve this case. Independent exact checking
accepts the full-hull Delaunay meshes, and every selected material cell has positive
orientation, but the retained boundary has pinches:

| Target | Selected cells | Edges with four boundary faces | Endpoint roles |
|---:|---:|---:|---|
| 1,000 | 2,968 | 29 | All join two fixed corners |
| 8,192 | 37,487 | 10 | Nine corner/corner; one surface/corner |

These are rejected candidates, not usable output meshes. A separate 120-degree,
four-subdivision 1K run succeeds, but its 8K counterpart fails; a single successful
parameter choice does not establish a universal default. The explicit 8K smooth
configuration in the control table still succeeds.

[#263](https://github.com/alecjacobson/warpower/issues/263) remains open for the
complete default workflow, feature sampling and recovery around fixed corners.
This change fixes device budgeting without weakening geometry checks or changing
the feature policy. Compact commands, numerical reports, independent checks and
raw-file hashes are in [experiments/initial-corner-budget](../experiments/initial-corner-budget/).
