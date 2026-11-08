# Captured variational tetrahedral meshing

`VariationalMeshingWorkspace` composes initial sampling, progressive refinement,
global ODT/boundary fitting, selective optimization, boundary jitter, interior
teleportation, and material extraction in one reusable device graph. It accepts
a closed oriented `SurfaceDomain` and returns a tetrahedral approximation of its
material, including nonconvex shapes, cavities, and disconnected components.
Its construction allocates storage and compiles/captures kernels; `run()` performs
the meshing decisions on the GPU. The implementation uses Warp kernels throughout.

```python
import warpower
import warp as wp

# vertices: float32 (V,3), faces: int32 (F,3), consistently outward oriented.
domain = warpower.SurfaceDomain(vertices, faces, surface_subdivisions=4)
target = wp.full(1, 1000, dtype=int, device=domain.device)
seed = wp.full(1, 2026, dtype=int, device=domain.device)
mesher = warpower.VariationalMeshingWorkspace(
    domain, capacity=2000, target_count=target, initial_count=256, seed=seed
)

with wp.ScopedCapture(device=domain.device) as capture:
    # An upstream kernel may change domain.points, target and seed here.
    domain.refit()
    mesher.run()
    # Downstream kernels read mesher.error and the device output counts.

wp.capture_launch(capture.graph)
```

For a fixed domain, `mesher.run()` also launches its own prebuilt graph. Each run
starts from fresh samples; it does not continue optimizing the preceding output.
Repeated identical input/seed specifies the same sampling stream. Floating-point
parallel reductions can differ by roundoff. The input surface connectivity is
fixed for the lifetime of `SurfaceDomain`; refit updates positions and derived
features. Refit an explicit sizing field before the mesher if its samples changed.

Each fresh sample generation resets the core's reconstruction preference. Strategy
reuse continues within that optimization trajectory; a difficult final state from
the preceding solve does not force its expensive strategy onto new initial sites.
This policy belongs to the full mesher. Independently used `DelaunayWorkspace`
keeps its existing strategy caching.

Keep the workspace, domain, sizing providers, and caller-owned input arrays alive
as long as any captured graph uses them. A workspace owns mutable scratch and
must not execute concurrently with itself. Serialize its runs on one stream or
with events; use separate workspaces for concurrent solves. `run(stream=...)`
requires a stream on the domain device. Shared domains/fields must not be refitted
while another solve reads them. Construction is a host setup operation and must
finish before entering the enclosing capture. There is no CPU geometry fallback.

The independent [diagram and full-hull routines](architecture.md),
[`DelaunayWorkspace`](delaunay-recovery.md), [ODT optimizer](odt.md),
[boundary fitter](boundary.md), and [extractor](extraction.md) remain callable
without constructing this mesher. Importing or using them does not load the new
scheduler or add meshing stages to their graphs.

## Inputs and budgets

`capacity` is the maximum number of sites. `target_count` is an integer or a
contiguous device `int32[1]`, defaulting to capacity. `initial_count` is a setup
integer, defaulting to `min(256, target)` for a static target and
`min(256, capacity)` for a device target. Every replay requires
`4 <= initial_count <= target <= capacity` and enough initial sites for all
current corner anchors. Increasing the device target does not allocate storage.
The target counts sites; some full-hull sites can be unused by retained material
tetrahedra. Consumers should use connectivity to identify the material vertices.

Uniform sizing is the default. Supply a [`SizingField`](sizing.md) for grading, or
set `automatic_sizing=True` to own a pole-based sizing provider that is refitted
on each replay. These choices are mutually exclusive. The owned provider defaults
to `automatic_control_subdivisions=0` (input surface vertices as control sites);
its accuracy depends on those samples. For more control, construct an independent
`AutomaticSizingWorkspace`, refit it in the enclosing graph, and pass its field.
Automatic sizing can dominate time on symmetric control clouds requiring symbolic
recovery; it is not a free preprocessing operation.

| Option | Default | Meaning |
|---|---:|---|
| `growth_iterations` | 5 | Global fitting iterations before each insertion batch |
| `refinement_fraction` | 0.5 | At most this fraction of the active count inserted per batch |
| `max_growth_rounds` | `ceil(log(capacity/initial)/log(1+fraction))+4` | Bounded growth attempts, including retries after stagnation |
| `polish_iterations` | 20 | Global fitting budget after reaching the target and per recovery round |
| `max_recovery_rounds` | 3 | Additional fitting rounds after an invalid or topologically mismatched extraction |
| `selective_iterations` | 10 | Selective optimization budget per quality round |
| `quality_rounds` | 4 | Maximum selective/jitter/interior rounds; zero omits all three stages and their storage |
| `quality_threshold` | 0.3 | Radius-ratio threshold selecting poor elements |
| `local_quality_retries` | 0 | Additional [local rollback trials](local-quality-backtracking.md) per selective step; opt-in, shape-dependent tradeoff |
| `jitter_attempts` | 8 | Bounded boundary proposal attempts per round |
| `extraction_tolerance` | 0.4 | Circumcenter thickening tolerance, `distance/radius` |
| `containment_recovery` | `True` | Bounded addition of proven wholly material cells around bad boundary edges |
| `exterior_recovery` | `True` | Transactional removal of exterior-center tolerance cells, with vertex/feature and volume guards |
| `sampling_resolution` | 32 | Proposal grid resolution per axis |
| `sampling_max_attempts` | 4096 | Rejection-sampling budget per initial site |
| `seed` | 2026 | Signed int32 or device int32[1]; caller-owned seed is never incremented |

`tetrahedra_capacity`, `tetrahedra_cell_capacity` (default 1024), and `chunk_size`
configure the underlying reconstruction reserves. Allocations scale with capacity,
not the current target. Iteration and round budgets are setup constants; loop
termination and stage selection use device counters and CUDA conditional nodes.
Zero polish iterations disable additional-fitting recovery. The independent
[containment repair](containment-recovery.md) can still resolve supported bad
boundary edges without moving sites.

Refinement ranks interior Voronoi centers and uses certified transactions. A failed
batch that restored a valid source retries a single insertion, then resumes bounded
growth. Failure to reach the target is explicit. Initial extraction must be
nonempty and locally manifold, and its component count, shell count, cavity count,
and aggregate boundary Euler characteristic must match the input surface.
Additional global fitting can repair a coarse invalid boundary. It never silently
exceeds the requested site count.

Before additional fitting, optional containment recovery examines excluded
full-hull cells around nonmanifold boundary edges and adds only cells proved
wholly in material. Every pass reruns the topology gate. Once repair is used,
quality guards evaluate the repaired subcomplex consistently; a fresh solve
resets this mode. The independent extractor keeps its original rule by default.
If bad edges remain after the additional-fitting budget, [exterior-band recovery](exterior-band-recovery.md) can
remove exterior-center tolerance cells while preserving referenced vertices and
feature-incident cells. It limits volume change and restores the selection on
failure. This policy is an approximation; it does not prove whole-cell exterior
membership. Both policies can be disabled independently.

Each optional quality stage starts with a valid-mesh checkpoint. A stage error or
invalid extracted boundary restores positions, connectivity, counts, statistics,
and boundary state entirely on device. A successfully restored rejected heuristic
does not invalidate the preceding mesh. Failure to restore gates all outputs.
Quality rounds stop when none of their stages makes progress or the budget expires.

## Device outputs

All arrays below have fixed addresses. Read valid prefixes only after checking
`error[0] == 0` or `valid[0] == 1` **on device** in a captured consumer.

| Array | Count or interpretation |
|---|---|
| `points` | `point_count[0]` sites; float32 vectors |
| `full_tetrahedra` | `full_tetrahedra_count[0]` positively oriented full-hull Delaunay tetrahedra |
| `tetrahedra` | `tetrahedra_count[0]` retained material tetrahedra |
| `boundary_faces` | `boundary_count[0]` outward oriented material triangles |
| `radius_ratio` | One `3*inradius/circumradius` value per retained tetrahedron |
| `counts` | The four counts above, in that order |
| `topology` | Material components, boundary shells, cavities, sum of shell Euler characteristics |
| `metrics` | Retained volume, input material volume, their signed difference, minimum radius ratio, directed boundary RMS distance |

Individual scalar views expose `component_count`, `shell_count`, `cavity_count`,
`euler`, `volume`, `domain_volume`, `volume_difference`, `minimum_radius_ratio`, and
`surface_rms_distance`. On failure, all published counts, topology and metrics
are zero. Connectivity/point buffers and child workspace scratch may contain
earlier data and must not be read as a published result. `active_count` is an
internal progress count; use the gated public `point_count` in consumers.

The RMS is sampled from the output boundary to the input surface. It is not a
two-sided Hausdorff bound. Input/output topology invariants catch missing whole
components, cavities and many handle changes, but do not prove a homeomorphism.
Neither those checks nor `valid` guarantee a requested approximation tolerance
or minimum element quality. Budgets, quadrature, sizing and input resolution
affect those measurements. Corners stay anchored; crease and surface membership
evolves during fitting. The boundary approximates the input rather than preserving
every input facet, as in the variational method.

## Status and diagnostics

`status` is 0 after success with quality rounds disabled, 1 after exhausting a
quality budget while making progress, 2 after quality stagnation, or 3 on error.
Statuses 1 and 2 still publish a valid mesh; neither asserts optimal quality.

`error` is a bitmask:

| Bit | Meaning |
|---:|---|
| 1 | Invalid domain/input topology, target, or initial corner budget |
| 2 | Invalid sizing or failed owned automatic sizing |
| 4 | Failed initial sampling |
| 8 | Failed global fitting/reconstruction |
| 16 | Fatal refinement or refinement restoration failure |
| 32 | Growth budget exhausted before the requested target |
| 64 | Nonempty manifold extraction with matching topology was not obtained within the recovery budget |
| 128 | Failed restoration of an optional quality stage |

`rounds` reports growth/recovery/quality rounds. `steps` accumulates accepted global
and selective optimizer steps; steps subsequently undone by a scheduler checkpoint
restoration remain included. `moves` counts committed jitter and interior moves.
`refinement_rejections` and `quality_rejections` count rejected transactions/stages.
`last_errors` holds the last sizing, sampling, core, refinement, jitter and interior
error codes; codes can remain nonzero after a successful fallback. Consult the
child workspaces for their meaning. `extraction_reason` uses bits 1 for extraction
error/invalid manifoldness, 2 for an empty mesh, and 4 for a topology mismatch;
`last_bad_extraction` retains the most recent quality-stage rejection reason.
Every replay resets public status and can recover after corrected input.

## Reproduction

```bash
python -m benchmarks.mesher --case torus --n 1000 --initial 256 --reference \
  --output results/local/mesher-torus.json --save-mesh results/local/mesher-torus.npz
python -m benchmarks.mesher --case cube --n 1000 --initial 1000 --quality-rounds 0 \
  --output results/local/mesher-fixed-budget.json
python -m benchmarks.mesher_loop --dot results/local/mesher-loop.dot \
  --output results/local/mesher-loop.json
python -m benchmarks.mesh_quality results/local/mesher-torus.npz \
  --output results/local/mesher-torus-independent.json
```

The first benchmark includes domain refit/validation, sizing when requested,
sampling, growth, optimization, quality stages, extraction and diagnostics.
It separately reports setup, first replay, warmed replay samples, allocator
usage and CPU quota counters. Setup timing depends on the compilation cache;
the reported pool delta is not a complete device/process memory peak.
`--reference` independently checks full-hull Delaunay cells and material topology
outside the timed region. `--initial N --n N` compares a fixed initial budget to
progressive growth. Compare quality and fit as well as time. The measured
[progressive/fixed-site schedule comparison](quality-frontier.md) includes
quality distributions, fit, alternate seeds and domains; it does not change defaults.

`benchmarks.mesh_quality` performs independent NumPy circumcenter/quality solves,
face/edge incidence checks, dihedral-angle measurements, and brute-force distance
queries in both boundary directions. These CPU measurements are validation tools
outside the mesher and outside timed replay, never a geometry fallback.

The initial [1,000-site torus result](../results/validation/vtm/mesher-torus1000.json)
contains 4,225 retained tetrahedra and the expected single genus-one boundary.
The [independent measurements](../results/validation/vtm/mesher-torus1000-independent.json)
give minimum/median radius ratio 0.0929/0.8744, minimum dihedral angle 3.06 degrees,
and volume error -5.09%. The two-sided sampled RMS is 0.256% of the input bounding
box diagonal. These expose remaining quality/fit work; they are not a universal
quality guarantee. A subsequent [uncontended replay measurement](../results/validation/vtm/mesher-torus1000-clean.json)
on L40 takes a median **6.031 seconds** over five samples, including domain refit
and the complete meshing pipeline. Both first-run and warmed CPU quota windows
are unthrottled. It reproduces the same counts and quality. The earlier 6.06-second
run had throttling in its combined initial/warm window and is retained only as
diagnostic evidence. Neither run establishes a CPU speedup.

These are historical implementation measurements. The
[surface-query launch comparison](query-launches.md),
[reconstruction/projection tuning](reconstruction-launches.md) and
[same-input cell-proof reuse](completion-proofs.md), together with
[current hull-support reuse](hull-support-reuse.md), record later complete-solve
timings, independent mesh comparisons and native capture audits.
The [extension of certified fragment reuse through 4,096 sites](large-fragment-reuse.md)
reduces complete 4k torus and Armadillo solves by 1.82–2.96× versus the preceding
GPU implementation; small default and standalone API timings remain stable.
The subsequent [analytic predicate filters](analytic-facet-search.md) reduce the
complete 4k torus from 3.785 to 2.825 s and the 1k torus from 824 to 737 ms,
with matching connectivity and iteration work. Standalone timings stay within
0.8% of controls; the uncertain and symbolic predicate policies are unchanged.
The [prepared point-query filter](prepared-point-queries.md) further reduces
the 4k torus to 2.640 s and the 1k torus to 717 ms, with unchanged storage and
launch scheduling. Final standalone comparisons stay within 0.4% of controls;
full hull, material, boundary and iteration-work checks remain unchanged.
The subsequent [grouped seed queries](grouped-seed-queries.md) reduce the 4k
torus to 2.401 s and the 1k torus to 704 ms, with fixed extra group storage.
Final meshes retain the same connectivity and work; standalone comparison
differences remain below 1% after retaining and pooling the fitted-hand repeats.

The complete-mesher CPU speedup target remains open in
[#40](https://github.com/alecjacobson/warpower/issues/40); the existing Paragram
diagram speedup is a different measured operation.

## Validation scope

The [362-test suite](../results/validation/vtm/mesher-full-tests.txt) passes,
including cavities, disconnected components, exact corners, explicit/automatic
sizing, growth retries, coarse graded extraction recovery, forced stage failures
and later valid replays. An additional [outer-loop test](../results/validation/vtm/mesher-outer-loop-test.txt)
executes producer/mesher/consumer iterations inside a device-controlled
`capture_while`, including invalid input and recovery, with Warp array allocation,
host readback and synchronization forbidden during capture and submission.

The [native five-phase audit](../results/validation/vtm/mesher-device-audit.json)
records 36,129 kernel activities. Its graph has 6,352 kernel, 1,894 memset,
770 conditional and 420 copy nodes; every graph copy is device-to-device.
There are no host/callback or allocation/free nodes or observed host transfers.
Nsight omits some graph copy activity records, so all DOT copy directions are
checked too. The [installed-wheel smoke test](../results/validation/vtm/mesher-wheel.json)
verifies source equality, captured count changes, invalid-target gating and
recovery from outside the source checkout.

Two [full-memory/lifetime cases](../results/validation/vtm/mesher-lifetime.txt)
and two [pooled-memory cases](../results/validation/vtm/mesher-pooled-memcheck.txt)
pass with zero sanitizer errors. They exercise rejected optional stages, failed
restoration and later recovery, bounded topology-recovery exhaustion, and a
failed insertion batch followed by successful single-site fallback. They use
32-byte padding and the [diagnostic stack/allocator settings](cuda-validation.md);
production context settings are unchanged.

The [memory-check scope record](../results/validation/vtm/mesher-memory-scope.json)
also preserves two interrupted broader attempts. The changing-input lifetime
matrix was stopped after more than 17 minutes with one completed case; the
graded pooled matrix was stopped after more than 12 minutes with two completed
cases. Both reported zero errors before termination, but neither is counted as
a completed matrix. The original graded recovery and all changing-input phases
pass the normal tests; the focused memory checks cover their scheduler branches.

The [kernel profile](../results/validation/vtm/mesher-loop-profile.json) identifies
202 symbolic cell reconstructions taking 16.30 seconds across the profiled
phases. [#81](https://github.com/alecjacobson/warpower/issues/81) investigates whether
the cached reconstruction preference should be refreshed when optimizer geometry
changes. Current results remain certified on the current input; this is a cost
question, not permission to weaken the predicate or topology checks.

## Reconstruction policy measurements

The five-phase changing-input experiment measures three valid small solves and
two rejected inputs per cycle. Three measured cycles on L40 give:

| Policy | Median full cycle |
|---|---:|
| [Retain preference across generations](../results/validation/vtm/recovery-policy-cached.json) | 17.312 s |
| [Reset once per generation](../results/validation/vtm/recovery-policy-reset.json), the new full-mesher policy | **11.689 s** |
| [Disable caching throughout](../results/validation/vtm/recovery-policy-uncached.json) | 30.027 s |
| [Probe ordinary construction before every preferred symbolic build](../results/validation/vtm/recovery-policy-probe.json) | 30.867 s |

The reset is **1.48x faster** on this loop. All runs are unthrottled and have
[identical final points and canonical connectivity](../results/validation/vtm/recovery-policy-equivalence.json),
matching phase counts/errors, and independent geometry validation. A separate
[paired 1,000-site torus comparison](../results/validation/vtm/mesher-reset-paired-torus1000.json)
alternates warmed historical and reset graphs: medians are 6.030 and 6.048 seconds
(+0.29%, within the observed timing variation), with identical points and cells.
These measurements do not establish the complete-mesher CPU speedup target.

```bash
python -m benchmarks.recovery_policy --policy reset --repeats 3 --reference \
  --output results/local/recovery-policy-reset.json
# Other policies: cached, uncached, probe. The harness explicitly reproduces
# the historical generation entry so the baseline is independent of the default.
python -m benchmarks.mesher_policy --repeats 5 \
  --output results/local/mesher-policy-torus.json
```

The reset change passes all [17 scheduler tests](../results/validation/vtm/mesher-reset-tests.txt).
Its [native audit](../results/validation/vtm/mesher-reset-device-audit.json) records
46,550 kernel activities across the five phases; the graph adds one memset node,
with no host/callback or allocation/free nodes and only device-to-device copies.
The [isolation check](../results/validation/vtm/recovery-reset-isolation.json)
confirms every numerical kernel and all other runtime modules are unchanged.
