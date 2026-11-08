"""Write the quadrature report from completed production qualification records."""
import json
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm'
def read(name):return json.loads((w/name).read_text())
source=read('surface-sample-production-source-v2.json')
proof=read('surface-sample-final-v1-independent.json');assert proof['complete']
qualified=read('surface-sample-qualified-v1.json');assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
memory=read('surface-sample-memory-v1-commands.json');assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
uniform=proof['cases']['armadillo-uniform8192-recovery']['timing'];graded=proof['cases']['armadillo-graded8192-recovery']['timing']
reduction=lambda t:100*(1-t['after_ms']/t['before_ms'])
text=f'''# Parallel boundary surface-error quadrature

One query per thread reduces complete uniform and graded 8K scanned Armadillo
replay time by **{reduction(uniform):.2f}% and {reduction(graded):.2f}%** in the final
paired measurements. The independent power-diagram and Delaunay routines retain
their implementations and callable APIs. This is an improvement between Warp
revisions; the comparable-quality native CPU outperformance goal remains open.

## Implementation

The extractor previously ran three nearest-surface queries sequentially in each
boundary-face thread. The current kernel maps thread `j` to face `j // 3` and
quadrature sample `j % 3`. One launch covers three times the reserved face capacity;
the current device count and error flag guard every access to inactive faces.
The three-point quadrature, area weights, double-precision closest-point
traversal, error bits, integral addition and maximum reduction are unchanged.

Only this kernel and its launch dimension change in production. The selection
extractor's status-only path is unchanged. There are no new persistent arrays,
host count reads, replay allocations, or extra launches. The arithmetic AST
matches the private measured kernel, while the test reference matches the old
sequential loop. The source audit binds {len(source['source_sha256'])} files, with
263 unchanged and changes confined to `_extraction.py` and two test files.

Atomic addition order changes, so the integral is not promised bitwise identical.
The initial instrumented comparisons preserve each nearest hit's point, face,
squared distance and weighted contribution exactly across the tested caller
contraction settings and scales. Static integral differences are a few parts in
10^15; maximum distances are bitwise equal. Full trajectories and independent
mesh quality receive separate checks below.

## Complete solves

NVIDIA L40, Warp 1.19.0+cu12; baseline `{source['baseline']}` and runtime
`{source['runtime']}`. Each warmed replay starts a fresh complete solve, including
refit, sampling, growth, fitting, quality work, recovery and extraction. Setup,
upload and offline validation are outside replay; setup and first-call results
remain in the raw records. Mesher-only constructor observations and benchmark
setup observations have distinct scopes, identified in the comparison report.
All 32 final complete/standalone jobs have clean first and timed quota windows.

| Complete example | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
'''
for name,row in proof['cases'].items():
    t=row['timing'];text+=f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {reduction(t):.2f}% |\n"
text+='''
These measurements are not a universal speedup claim. Per-case neutral or slower
results remain visible. Input arrays, site roles, material connectivity, oriented
boundaries and recorded solver work match for all pairs. The independently
measured minimum radius ratios agree within 1e-12.

'''
variations=[]
for name,row in proof['cases'].items():
    c=row['comparison']
    if not c['points_bitwise_equal'] or not c['full_oriented_tets_equal']:
        variations.append(f"- {name}: {c['changed_scalar_coordinates']} changed scalar coordinates, maximum absolute difference {c['maximum_coordinate_difference']:.17g}; equal oriented full tetrahedra: {c['full_oriented_tets_equal']}.")
if variations:text+='The following coordinate/full-hull differences are explicit; both meshes receive valid current-geometry proofs:\n\n'+'\n'.join(variations)+'\n\n'
else:text+='All eleven pairs also preserve coordinates bitwise and oriented full-hull tetrahedra exactly.\n\n'
text+=f'''All 22 final meshes have valid independent full-hull empty-sphere/orientation/
coverage checks, positive material geometry, closed edge topology, single-cycle
boundary vertex links and zero checked boundary self-intersections. Of these,
{proof['exact_geometry_transfers']} reuse a previous independent proof only after
exact comparison of every array, oriented connectivity and aligned stored quality.
Other meshes receive fresh native and NumPy checks. All fourteen current gallery
VTUs and boundary OFF exports round-trip exactly. Surface-distance measurements
use quadrature and are not Hausdorff certificates. The scanned domain keeps its
[original provenance](scanned-armadillo.md); it is not claimed as an original
Alliez-paper input or relicensed under MIT.

## Independently callable routines

| Routine/input | Before, ms | After, ms | Change |
|---|---:|---:|---:|
'''
for name,t in proof['standalone'].items():text+=f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {100*t['relative_change']:+.2f}% |\n"
text+='''
All recorded inputs, counts, errors and diagnostic fields match. These small
differences do not establish a standalone speed gain. The README retains
installation instructions and the earned Paragram flagship.

## Static experiment and qualification

The static batches use independently validated final meshes and the full reserved
boundary capacity (32 times the site budget). Resets and padded launch work are
included; setup and host checks are excluded. Each row has twenty warmed GPU
samples. The 32-thread candidate was selected before the complete pilot.

| Final mesh | Original, ms | Samples/32, ms | Samples/64, ms | Samples/128, ms |
|---|---:|---:|---:|---:|
'''
for name in ['torus1000','torus16384','armadillo-uniform8192','armadillo-graded8192']:
    r=read('surface-sample-static-v1/'+name+'.json');text+='| '+name+' | '+' | '.join(f"{v['median_ms']:.3f}" for v in r['rows'])+' |\n'
text+='''
The private compiled kernel reports 78 registers/thread versus 96 for the original,
with the same 3,296 local bytes/thread. Those function properties do not measure
occupancy or explain a full-solve gain by themselves. The four pilot pairs retain
all coordinates, oriented cells, roles and work counts, with zero additional
workspace pool bytes.

The production suite passes all 1,387 tests. The 41 new cases cover both caller
contraction settings, six shapes at three scales, poisoned inactive capacity,
empty/failed queries, live counts and capture on another stream. Tests restore
module options after each case and synchronize setup before changing streams.
The unchanged full 8K five-phase loop covers bounded recovery exhaustion, success,
invalid target, collapsed domain and recovery, with stable buffer addresses and
public failure counts cleared.

'''
native={k:read(f'surface-sample-{k}-native-v1-audit.json') for k in ['uniform','loop']}
text+=f'''Native captures observe {native['uniform']['gpu_kernel_activities']:,} full-solve
and {native['loop']['gpu_kernel_activities']:,} loop kernel activities. The retained
CUDA activity and graph-DOT audits contain only device-to-device copies and no
host callbacks or replay allocation/free nodes. Each run has one outer graph
launch and a final synchronization; first loop instantiation/upload, where
present, is reported in the raw API list. These profiled times are diagnostic,
not substitutes for clean timing samples.

Four isolated sanitizer jobs pass with zero errors: quadrature contracts and the
full recovery loop, each with padding and with pool-disabled allocation-lifetime
checking. The loop jobs took {next(r['wall_seconds'] for r in memory if r['name']=='loop' and r['mode']=='padded'):.1f} s
and {next(r['wall_seconds'] for r in memory if r['name']=='loop' and r['mode']=='lifetime'):.1f} s.
The existing diagnostic stack allowance is recorded by the runner; production
benchmarks do not use that override. Ordinary, native and sanitizer loop frames
match independently certified reference geometry and consumed diagnostics.

The record retains an interrupted pilot driver with an export-loop naming bug,
two offline checker setup errors, and an interrupted first suite run before
test-option cleanup was corrected. None is counted as passing or as a numerical
failure. Original scripts, logs, partial outputs and corrections remain intact.

The [manifest](../results/validation/vtm/surface-sample-quadrature/manifest.json)
binds raw timings, complete meshes, source versions, failures, native traces,
memory logs and independent proof dependencies. This completes
[#252](https://github.com/alecjacobson/warpower/issues/252).
[#253](https://github.com/alecjacobson/warpower/issues/253) tracks a separate
nearest-traversal hypothesis; it is not part of this implementation.
'''
p=root/'docs/surface-sample-quadrature.md';assert not p.exists();p.write_text(text)
p=root/'docs/ordered-surface-queries.md';s=p.read_text();s+='\n[Parallel boundary quadrature](surface-sample-quadrature.md) subsequently reduces\ncomplete scanned-example replay time without changing classifier ordering or\nadding persistent arrays.\n';p.write_text(s)
print('Report generated from completed qualification records.')
