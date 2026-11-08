"""Write final documentation only after production and independent gates complete."""
import json
from pathlib import Path
w=Path('.work/vtm');out='results/validation/vtm/larger-active-prefixes'
def read(n):return json.loads((w/n).read_text())
r=read('larger-prefix-final-v1-independent.json');assert r['complete'] and len(r['cases'])==11 and len(r['standalone'])==5
assert read('larger-prefix-grid-v1-independent.json')['complete']
assert read('larger-prefix-native-loop-independent-v1.json')['complete']
assert read('larger-prefix-full-suite-v1.json')['tests']==1292
mem=read('larger-prefix-memory-v1-commands.json');assert len(mem)==28 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in mem)
full=read('larger-prefix-full-native-v1-audit.json');loop=read('larger-prefix-loop-native-v1-audit.json')
labels={'reserved16384-target4096':'Torus target 4,096; reserved 16,384','reserved8192-target4096':'Torus target 4,096; reserved 8,192','reserved16384-target8192':'Torus target 8,192; reserved 16,384','torus8192-progressive':'Torus 8,192; progressive','graded-torus16384':'Graded torus 16,384; progressive','torus16384-fixed32':'Torus 16,384; fixed 32','torus8192-fixed32':'Torus 8,192; fixed 32','torus4096':'Torus 4,096','torus1000':'Torus 1,000'}
def row(name,x):return f"| {name} | {x['before_ms']:,.3f} | {x['after_ms']:,.3f} | {x['before_ms']/x['after_ms']:.3f}× |"
timing='\n'.join(row(label,r['cases'][name]) for name,label in labels.items());standalone='\n'.join(row(name,x) for name,x in r['standalone'].items())
mem_rows=[]
for name in ['reserved16384-target4096','torus8192-progressive','torus16384-fixed32']:
 x=r['cases'][name];p=x['persistent_workspace_pool_bytes'];s=x['setup_seconds']
 mem_rows.append(f"| {labels[name]} | {p['before']:,} | {p['after']:,} | {s['before']:.3f} → {s['after']:.3f} |")
memory='\n'.join(mem_rows)
for name,extra in [('reserved16384-target4096',1853169562),('torus16384-fixed32',1853169562),('torus8192-progressive',894484877)]:
 p=r['cases'][name]['persistent_workspace_pool_bytes'];assert p['after']-p['before']==extra
for name in ['uniform-armadillo8192','armadillo8192-graded-q2']:
 x=r['cases'][name];assert x['before_valid']==x['after_valid']==0 and all(x['failed_state_equal'].values())
for name in labels:
 x=r['cases'][name];assert all(x['work_equal'].values()) and all(x['comparison'][k] for k in ['points_bitwise_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal','roles_equal'])
g=read('larger-prefix-grid-v1-independent.json');grid_before=g['cases']['before']['median_ms'];grid_after=g['cases']['after']['median_ms']
cross=read('larger-prefix-current-reservation-comparison-v1.json');assert cross['complete']
x4=cross['cases']['reserved16384-target4096'];x8=cross['cases']['reserved16384-target8192']
s=f'''# Larger device-selected active prefixes

VTM now preallocates fitting 4,096- and 8,192-site constructors alongside its
existing 1,024/2,048 routes and full constructor. The device active count selects
the smallest fitting branch. A 16K reservation can therefore run a 4K target
through the 4K constructor while keeping stable public outputs for later growth.
The tradeoff is additional persistent memory and setup; each child has separate
storage. This is not the scratch-sharing work in [#122](https://github.com/alecjacobson/warpower/issues/122).

## Complete solves and costs

Production `670bd3b` is compared with `df74493` on NVIDIA L40 / Warp 1.19.0.
Nine fresh interleaved complete-solve pairs record three measured replays each,
including domain refit, fresh sampling, fitting, refinement/quality and extraction.
Setup, compilation, upload and offline checks are excluded from replay. The
[primary comparison](../{out}/larger-prefix-final-v1-independent.json) preserves
samples, setup, first replay, workspace memory and independent outputs. Every
initial and measured primary timing window passes the zero-CPU-quota-throttling
gate. These are gains over the preceding GPU implementation; the
[wide comparable-quality CPU goal](fixed-site-current.md) remains unmet.

| Complete solve | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
{timing}

All eighteen successful saved meshes independently pass original-coordinate
full-hull/orientation/empty-sphere, material quality/topology, boundary vertex-link
and boundary self-intersection checks. Graded exports round-trip. Every pair has
bitwise-identical coordinates, canonical full/material cells, oriented boundary,
roles and solver work. Routing counts change intentionally and remain in the raw
JSON; reduced constructor capacity does not mean fewer optimizer iterations.

A [cross-reservation check](../{out}/larger-prefix-current-reservation-comparison-v1.json)
also finds identical geometry and work across the saved post-change reservations.
Using the primary medians, a 4K target reserved at 16K records
{x4['reserved_over_dedicated']:.3f}× its dedicated 4K time; the 8K target reserved
at 16K records {x8['reserved_over_dedicated']:.3f}× its dedicated 8K time. These reuse
the existing samples; no separate interleaved cross-reservation timing experiment
was performed.

| Solve | Before pool bytes | After pool bytes | Setup seconds |
| --- | ---: | ---: | ---: |
{memory}

These are persistent workspace pool bytes, not process peak memory. Both extra
children add **1,853,169,562 bytes** at reservation 16K; the 4K child adds
**894,484,877 bytes** at reservation 8K in these configurations. Even the fixed 16K
case pays for unused children because a device-controlled target may change in
a later graph invocation. Standalone diagram/Delaunay construction allocates
neither additional child. README retains its earned Paragram flagship.

## Prefix proof and ownership

VTM first copies active coordinates to separate mesh scratch and fills every
inactive slot with an exact copy of site zero. Taking a fitting prefix removes
only later duplicates: every distinct active position, canonical first owner
and original site ID remains. Each selected constructor still certifies the
complete current active convex hull with the existing original-coordinate and
symbolic predicates. [The proof note](../{out}/larger-prefix-proof-v1.md) states
these prerequisites; arbitrary unprepared buffers do not satisfy the contract.

The [VTM owner](../warpower/_vtm.py) supplies explicit increasing levels filtered
below its full reservation. The [private router](../warpower/_active_delaunay.py)
keeps its direct default of 1,024/2,048 and accepts explicit levels through 8,192.
The full constructor remains output owner and fallback. Every child has separate
retained-fragment storage; public result copies are DtoD. A fresh solve clears
fragments/preferences for every route. Reuse always re-establishes current-input
proofs after movement, growth, shrinkage, rejection or unrelated input.

All constructors, views and conditional graphs exist before capture. Selection,
execution, result copies and fallback stay on device. No new GPU kernel,
predicate, certificate, reset, publish or fallback method is introduced. An
[AST/source audit](../{out}/larger-prefix-production-source-audit-v2.json) binds the
production setup to the measured private prototype, verifies 148 source hashes
and checks 141 other runtime/benchmark/example files unchanged.

This proof concerns geometric equivalence. Existing parallel cell ordering and
floating reductions do not guarantee identical VTM trajectories on every possible
input. Strict equal-output/work checks are reported for the measured cases;
independent validity is checked separately.

## Tests and captured execution

The full suite passes **1,292 tests** and the focused routing suite passes
**47 tests**. Thirteen new cases cover captured 4,095/4,096/4,097 and
8,191/8,192/8,193 thresholds, growth/shrinkage, unrelated inputs, invalid counts,
nonfinite/flat data, recovery, exact ties, duplicate first owners at sparse
indices, one-ULP changes, scales 2^-60/1/2^60, exhausted outputs, forced child
failures and the non-reusing full route above 16K. Capture tests forbid host
allocation/readback/synchronization and use a non-default stream.

All **28 isolated memory jobs** pass padded-access and pool-disabled lifetime
checks with zero memcheck errors. These diagnostic processes use a 131,072-byte
CUDA stack for instrumentation; production execution settings are unchanged.
[Commands and logs](../{out}/larger-prefix-memory-v1-commands.json)
retain the individual results and 600-second process limits. Periodic 60-second
Python stack dumps are diagnostic snapshots, not termination or completion.
A terminated or failed job is never counted as a pass.

Native traces cover a warmed complete fixed 16K solve and a device-controlled
producer/mesher/consumer loop with reservation 16K. The loop runs targets 4K and 8K
with a domain deformation, rejects target 16,385 and a collapsed domain, then
recovers the original 4K solve. All three valid frames independently validate,
invalid frames publish zero outputs/consumed volume, output pointers stay stable,
and recovery matches exactly.

The [full audit](../{out}/larger-prefix-full-native-v1-audit.json) and
[loop audit](../{out}/larger-prefix-loop-native-v1-audit.json) record
{full['gpu_kernel_activities']:,} and {loop['gpu_kernel_activities']:,} kernel
activities. Captured nodes are device kernels, conditional work, memset and DtoD
copies. No host transfer, callback or explicit allocation/free occurs inside
measured execution. Every DOT copy node is checked because Nsight can omit copy
activity records. Each trace has one graph launch and external final
synchronization. The first loop launch can include deferred graph instantiation
and upload before that launch; this is not a claim of zero driver-internal host
work. Native/sanitizer timings are diagnostic rather than clean speed samples.

## Standalone APIs and remaining failures

| Callable workload | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
{standalone}

The standalone pairs record nine complete Delaunay replays and thirty diagram
runs, with matching inputs/counts/status/diagnostics. Their construction paths
are unchanged. Existing fitted-hand timing variability remains tracked in
[#235](https://github.com/alecjacobson/warpower/issues/235); no samples are dropped.
A retained 17³ grid guard measures **{grid_before:.6f} → {grid_after:.6f} ms** after
initial construction. Both outputs independently certify the complete hull and
physical quality and have identical canonical cells. This is retained-mesh
stress rather than a fresh VTM solve or CPU comparison.

Two additional fresh pairs retain the known uniform/graded 8K Armadillo rejection
cases. Both versions report error 64 with zero published outputs and matching
failed state/work. Their timings are not successful-mesh speedups. The extraction
problem remains [#228](https://github.com/alecjacobson/warpower/issues/228); no
quality or topology guard is weakened.

The [private decision](../{out}/larger-prefix-adoption-decision-v1.md) records the
earlier four clean pilot pairs, all eight independently validated pilot meshes,
twelve private correctness cases and the explicit memory/setup tradeoff.
[The manifest](../{out}/manifest.json) links exact sources, raw results, proofs,
independent validators, native traces and memory logs. Tracked in
[#230](https://github.com/alecjacobson/warpower/issues/230), supporting
[#29](https://github.com/alecjacobson/warpower/issues/29),
[#40](https://github.com/alecjacobson/warpower/issues/40) and
[#43](https://github.com/alecjacobson/warpower/issues/43).
'''
p=Path('docs/larger-active-prefixes.md');assert not p.exists();p.write_text(s)
print('Wrote report; review numerical claims, wording, native APIs and follow-up links before publication.')
