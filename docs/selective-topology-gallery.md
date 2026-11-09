# Local material rollback: broader gallery and graph qualification

The endpoint policy from the [initial pilot](selective-topology-rollback.md) is
measured at `313b3f9`, on production runtime `d7d6675` with shared clipping scratch.
No Warpower runtime files change in this experimental branch. The new driver
selector exercises the complete torus and changing-input loop programs too.
Production integration is separate, under #290; these data are not its gate.

## Complete solves

NVIDIA L40 / Warp 1.19.0, median five additional complete captured solves per
configuration. Setup, transfers, export and independent validation are separate;
every accepted window has full CPU quota. Every case uses the same inputs/options
within its pair, except whether the experimental endpoint policy is installed.

| Case | Baseline (ms) | Endpoints (ms) | Change | Local retries |
|---|---:|---:|---:|---:|
| torus16384-fixed32 | 2908.720 | 2915.188 | +0.222% | 0 |
| torus1000 | 487.124 | 490.505 | +0.694% | 0 |
| torus8192-fixed32 | 1240.446 | 1242.431 | +0.160% | 0 |
| torus8192-progressive | 1977.128 | 1979.291 | +0.109% | 0 |
| thin-hand-seed7 | 1224.814 | 1222.859 | -0.160% | 0 |
| cavity | 334.880 | 337.236 | +0.704% | 0 |
| disconnected | 477.227 | 479.589 | +0.495% | 0 |
| plate | 785.965 | 788.637 | +0.340% | 0 |
| armadillo-graded4096 | 2064.178 | 2069.790 | +0.272% | 0 |
| armadillo-graded8192-recovery | 3336.729 | 3330.671 | -0.182% | 0 |
| armadillo-uniform8192-recovery | 3459.277 | 3349.790 | -3.165% | 6 |
| armadillo-graded8192-native-mixed | 2305.967 | 2310.310 | +0.188% | 0 |
| armadillo-uniform8192-native-mixed | 2323.640 | 2220.155 | -4.454% | 6 |

Both uniform 8K scans exercise six local retries, 24 blocked-site events, 18 fresh
eligible failures and one acceptance immediately after a local retry. Accepted
selective steps increase from two to four with the reference backend and from
three to four with native/mixed queries. The other eleven cases never enter
the local retry path; their measured changes stay below 1% in magnitude. These
five-repeat observations do not establish a universal performance bound.

All 26 saved meshes pass independent exact original-coordinate full-hull CGAL,
material incidence, boundary vertex-link, exact boundary-intersection and gallery
VTU round-trip checks. The uniform scans retain the initial pilot's tradeoff:
minimum radius ratio drops from 0.03101918236 to 0.03100472151 (about 0.047%
relative), while mean/lower-tail quality and sampled surface fit improve slightly.
Full/material tetrahedron counts both decrease by two and internal connectivity
changes; roles and oriented boundary connectivity match. Maximum coordinate movement versus baseline is 0.0012862682.

The hand has a 6.938893903907228e-18 maximum coordinate difference and a different
independently valid full-hull triangulation, with equal material tetrahedra and
oriented boundary. Disconnected components differ by at most 4.656612873077393e-10
with equal canonical/oriented connectivity. All remaining site arrays and canonical
connectivity are identical. Full numerical metrics, including angle extrema,
volume error and sampled fit, are retained. There is no CPU speedup claim.

## Changing inputs and actual local retries

Thirteen focused contracts pass on this branch. Baseline, endpoint and vertex-star
policies also pass the same five-phase producer/consumer loop: valid target 96,
deformed target 128, invalid target, collapsed domain and valid recovery at 96.
Invalid phases publish zero counts/consumed volume. These ordinary runs include
two complete cycles through the phases.

A native reference-backend scan trace contains 92,114 kernel activities and
actually exercises six local retries. Its sites, roles, inputs, canonical full/
material connectivity, oriented boundary, work counts and local counters match
the independently checked timed mesh exactly. The separate five-phase native
loop has 42,830 kernel activities. Both audits show only device-to-device copies,
with no host callbacks or replay allocation APIs. DOT copies are checked because
CUPTI omits many graph copy activities. Optional native Mesh refit remains a
separate limitation; this audit uses the reference surface backend.

## Instrumentation scope

The full changing-input loop passes padded memory checking in 476.47 seconds.
The exercised 8K reference scan passes padded checking in 192.37 seconds and
use-after-free tracking with the pool disabled in 311.1 seconds. Both scan outputs
are exactly bound to the independently validated timed mesh and retain the same
six local retries. These instrumented durations are not performance measurements.

The separate full allocation-order loop reaches its 600-second bound without a
sanitizer summary or an invalid-global-access report. It is **incomplete, not
passed**. This timeout does not diagnose a memory bug or establish a hang; progress
controls and instrumentation cost remain under #155. The earlier focused 13-case
full-tracking tests are not substituted for this full-path result. Diagnostic
stack allowances are explicit in the commands and never change production setup.

The current evidence supports evaluating an opt-in endpoint policy, with the
quality tradeoff disclosed and default zero preserved. Production source, graph
ordering, memory scope and public performance still need their own qualification.
All raw files and failed/incomplete attempts are checksum-pinned in the
[evidence index](../experiments/selective-topology-gallery/).
