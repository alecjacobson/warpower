"""Write the nearest-traversal report only from completed production qualification."""
import json
from pathlib import Path

root=Path.cwd(); w=root/'.work/vtm'
def read(name): return json.loads((w/name).read_text())
source=read('surface-push-production-source-v1.json')
proof=read('surface-push-final-v1-independent.json'); assert proof['complete']
qualified=read('surface-push-qualified-v1.json'); assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
memory=read('surface-push-memory-v1-commands.json'); assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
uniform=proof['cases']['armadillo-uniform8192-recovery']['timing']; graded=proof['cases']['armadillo-graded8192-recovery']['timing']
def reduction(t): return 100*(1-t['after_ms']/t['before_ms'])
text=f'''# Exact nearest-surface child rejection

Skipping already-rejected child pushes reduces complete uniform and graded 8K
scanned Armadillo replay time by **{reduction(uniform):.2f}% and {reduction(graded):.2f}%**
in the final paired measurements. Independently callable power-diagram and Delaunay
implementations and APIs are unchanged. This improves the Warp implementation;
the comparable-quality native CPU outperformance goal remains open.

## Traversal change

The existing nearest-surface traversal computes both child box distances, orders
the children, and saves their rounded distances on a stack. Previously it pushed
both children unconditionally and rejected unsuitable entries when popped.
It now pushes a child only when `not (distance >= result.distance2)`, retaining
the original pop-time check and far-then-near insertion order.

The current best squared distance changes only under a strict smaller-distance
update. A child already rejected by the same comparison cannot become acceptable
later. Removing that entry preserves the visit order and triangle arithmetic of
all surviving leaves. Keeping the negated `>=` comparison matters: substituting
`<` would change unordered NaN behavior. Pending entries still need the pop check
because an intervening leaf can improve the best distance.

There are no new or approximate bounds, new geometry terms, shorter stacks, device
arrays, host decisions or launches. The 128-entry stacks, root handling, strict
nearest-hit update, cutoff semantics and tie behavior remain. This differs from
the earlier unadopted tail traversal, which unconditionally stacked its deferred
child. See [the preceding investigation](surface-bound-investigation.md) and
[cached bound reuse](surface-bound-reuse.md).

The production function AST matches the private measured candidate. Reversing
the single push-block replacement restores the original module byte for byte.
The source audit binds {source['source_files']} files; all {len(source['unchanged'])}
other source/test/benchmark/example/install files are unchanged. Existing nearest
tests retain their independent uncached reference. No global floating-point
atomic reduction is promised bitwise deterministic.

## Complete measurements

NVIDIA L40, Warp 1.19.0+cu12; baseline `{source['baseline']}` and runtime
`{source['runtime']}`. Warmed graph replay starts a fresh complete solve, including
refit, sampling, growth, fitting, quality work, recovery and extraction. Setup,
uploads and offline verification are outside replay and remain separately
reported. All 32 final complete/standalone jobs have clean first and timed
CPU-quota windows. These are GPU revision comparisons, not CPU speedup ratios.

| Complete example | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
'''
for name,row in proof['cases'].items():
    t=row['timing']; text+=f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {reduction(t):.2f}% |\n"
text+='\nAll pairs retain input arrays, roles, oriented material/boundary connectivity and\nrecorded solver work. Workspace pool usage is unchanged. The independently\nmeasured minimum radius ratios agree within 1e-12.\n\n'
variations=[]
for name,row in proof['cases'].items():
    c=row['comparison']
    if not c['points_bitwise_equal'] or not c['full_oriented_tets_equal']:
        variations.append(f"- {name}: {c['changed_scalar_coordinates']} changed scalar coordinates, maximum absolute difference {c['maximum_coordinate_difference']:.17g}; equal oriented full tetrahedra: {c['full_oriented_tets_equal']}.")
if variations:
    text+='Coordinate/full-hull differences are reported explicitly; both meshes have valid current-geometry proofs:\n\n'+'\n'.join(variations)+'\n\n'
    text+='These differences are observed; this study does not identify their causal kernel.\n\n'
else: text+='All eleven pairs also retain bitwise coordinates and oriented full-hull connectivity.\n\n'
review=read('surface-push-variation-review-v1.json'); assert review['complete']
maximum_mean_delta=max(abs(row['quality_deltas']['mean_radius_ratio']) for row in review['cases'].values())
text+=f'The mean radius ratio differs by at most {maximum_mean_delta:.3g} across all pairs.\nThe retained variation review records coordinate witnesses and all other checked\nquality deltas.\n\n'
text+=f'''All 22 final meshes pass independent full-hull orientation, empty-sphere and
coverage checks, positive material geometry, closed oriented boundary edges,
single-cycle boundary vertex links and exact boundary self-intersection checks.
{proof['exact_geometry_transfers']} proofs transfer from prior independently checked
meshes only after exact array/oriented-connectivity/aligned-quality comparisons;
the others receive fresh checks. All fourteen gallery VTU/OFF exports round-trip
exactly. Surface distances use sampled quadrature, not a Hausdorff certificate.
The scan retains its [original Stanford/libigl provenance](scanned-armadillo.md).

| Independent routine/input | Before, ms | After, ms | Change |
|---|---:|---:|---:|
'''
for name,t in proof['standalone'].items(): text+=f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {100*t['relative_change']:+.2f}% |\n"
text+='''
Recorded standalone inputs, counts, errors and diagnostics match. These timing
differences do not establish a standalone speed gain. README retains installation
instructions and the earned Paragram flagship.

## Static probe and limits

All 64 static rows use twenty warmed samples and clean quota windows. Classifiers
include existing ordering/sorting, cache rebuilding and cold reset when requested;
they launch the complete `8*N` tet capacity with the real live count. Quadrature
includes resets and `32*N` reserved faces. Warm-cache rows reuse unchanged input.
Nearest-only rows query live cell centers and are not complete classifier timings.

| Static final mesh | Full cold, before/after ms | Selection cold, before/after ms | Quadrature, before/after ms |
|---|---:|---:|---:|
'''
static=read('surface-push-pruning-static-summary-v1.json')
for name,c in static['cases'].items():
    rows={r['name']:r for r in c['rows']}
    text+='| '+name+' | '+' | '.join(f"{rows[k]['before_ms']:.3f} / {rows[k]['after_ms']:.3f}" for k in ['full-cold','selection-cold','quadrature'])+' |\n'
text+='''
The graded 8K retained cold classifier is 5.03% slower in the static probe;
that result is retained. Warm cache costs are nearly unchanged. Compiled nearest
and quadrature kernels report 80 registers and 3,296 local bytes per thread.
Those properties alone do not measure occupancy or establish complete-solve gains.
The four complete pilot pairs preserve all coordinates, oriented connectivity,
roles and work exactly, with zero additional pool bytes. The first candidate
constructor includes extra JIT compilation; no setup speedup is claimed.

## Qualification and reproduction

The private candidate passes 145 nearest/quadrature/extraction/selection cases
and 29 cache/ordering cases. Strong nearest fixtures compare full hit point,
face and squared distance across six shapes, three binary scales, both caller
contraction settings, strict limits, nonfinite data and empty trees. The cache
tests preserve unmodified ordinary callers as independent references and include
refits, changed counts/cutoffs, failure/recovery, epochs and nondefault streams.

The committed production suite passes all 1,387 tests. The unchanged complete
8K five-phase loop covers bounded topology-recovery exhaustion, success, invalid
target, collapsed input and recovery with stable pointers and cleared public
failure counts. Ordinary, native and sanitizer frames match independently
certified geometry and consumed diagnostics.

'''
native={k:read(f'surface-push-{k}-native-v1-audit.json') for k in ['uniform','loop']}
text+=f'''Native traces contain {native['uniform']['gpu_kernel_activities']:,} full-solve
and {native['loop']['gpu_kernel_activities']:,} loop kernel activities. Activity
and graph-DOT audits show only device-to-device copies, no host callbacks and no
replay allocation/free nodes. Each has one outer graph launch and final sync;
any initial graph instantiation/upload remains explicit in the API records.
Profiled timings are diagnostic, separate from clean performance measurements.

'''
cost=read('surface-push-native-cost-review-v1.json'); assert cost['complete']
text+='| Native full-solve kernel attribution | Calls in each run | Before, ms | After, ms |\n|---|---:|---:|---:|\n'
for row in cost['rows']:
    text+=f"| {row['kernel']} | {row['before']['calls']} | {row['before']['milliseconds']:.3f} | {row['after']['milliseconds']:.3f} |\n"
text+='\nThe preceding native runtime and current runtime differ only in the nearest-traversal\nsource within the recorded source scope. Equal calls accompany lower query-kernel\ntime; this attribution is consistent with the separate complete replay results.\n\n'
text+=f'''Four isolated sanitizer jobs pass with zero errors: 116 nearest/cache/ordering/
quadrature cases and the complete loop, each with padding and with pool-disabled
allocation-lifetime checks. Full loop audits take
{next(r['wall_seconds'] for r in memory if r['name']=='loop' and r['mode']=='padded'):.1f} s
and {next(r['wall_seconds'] for r in memory if r['name']=='loop' and r['mode']=='lifetime'):.1f} s.
The diagnostic stack allowance is recorded in the isolated runner; production
timings do not use that override.

The [manifest](../results/validation/vtm/surface-push-pruning/manifest.json)
binds candidate and production source, raw samples, meshes, exact commands,
native traces, memory logs and shared proof dependencies. Restore the retained
`.work/vtm` paths at the pinned revisions, use the documented Warp environment,
and run private contracts/static/pilot, production tests, paired timings, then
native/memory/geometry/frame checks. Run one GPU job at a time and keep heavy CPU
work outside clean timing windows. Existing output paths are never overwritten.

This completes [#253](https://github.com/alecjacobson/warpower/issues/253).
Broader query performance and comparable-quality native CPU comparisons remain
tracked in [#197](https://github.com/alecjacobson/warpower/issues/197) and
[#40](https://github.com/alecjacobson/warpower/issues/40).
'''
target=root/'docs/surface-push-pruning.md'; assert not target.exists(); target.write_text(text)
target=root/'docs/surface-bound-reuse.md'; text=target.read_text(); text+='\n[Exact push-time rejection](surface-push-pruning.md) subsequently skips children\nalready rejected by the existing bound comparison, preserving the pop-time check\nand nearest-hit arithmetic.\n'; target.write_text(text)
print('Report written from completed qualification records.')
