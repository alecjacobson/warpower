"""Write the report only from completed production and independent evidence."""
import json
from pathlib import Path
w=Path('.work/vtm')
def read(n):return json.loads((w/n).read_text())
r=read('prepared-new-cell-final-v2-independent.json');g=read('prepared-new-cell-grid-v1-independent.json');assert r['complete'] and g['complete']
profile=read('prepared-new-cell-full-native-v1-kernels.json');mem=read('prepared-new-cell-memory-v1-commands.json');assert len(mem)==28 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in mem)
full=read('prepared-new-cell-full-native-v1-audit.json');loop=read('prepared-new-cell-loop-native-v1-audit.json');assert read('prepared-new-cell-native-loop-independent-v1.json')['complete']
assert read('prepared-new-cell-full-suite-v1.json')['tests']==1279
labels={'torus16384-fixed32':'Uniform torus16,384, fixed32','torus8192-progressive':'Uniform torus8,192, progressive','torus8192-fixed32':'Uniform torus8,192, fixed32','reserved16384-target4096':'Torus target4,096, reserved16,384','graded-torus16384':'Graded torus16,384, progressive','torus4096':'Uniform torus4,096','torus1000':'Uniform torus1,000'}
labels={k:v.replace('torus','torus ').replace('fixed32','fixed 32').replace('target4','target 4').replace('reserved16','reserved 16') for k,v in labels.items()}
def row(name,x):return f"| {name} | {x['before_ms']:,.3f} | {x['after_ms']:,.3f} | {100*(x['after_ms']/x['before_ms']-1):+.2f}% |"
table='\n'.join(row(labels[name],r['cases'][name]) for name in labels);standalone='\n'.join(row(name,x) for name,x in r['standalone'].items())
names=['prepare_unproved','check_prepared_unproved','strict_check','exact_check','retained_cached','search_unproved_faces','record_face_support'];selected=[]
for name in names:
 matches=[x for x in profile['kernels'] if x['kernel'].startswith(name+'_')];assert len(matches)==1;selected.append(dict(matches[0],label=name))
kernels='\n'.join(f"| {x['label']} | {x['calls']} | {x['registers']} | {x['milliseconds']:,.3f} |" for x in selected)
for x in r['cases'].values():
 if 'persistent_workspace_pool_bytes' in x:assert x['persistent_workspace_pool_bytes']['before']==x['persistent_workspace_pool_bytes']['after']
pools='\n'.join(f"| {n:,} | {r['cases'][name]['persistent_workspace_pool_bytes']['before']:,} | {r['cases'][name]['persistent_workspace_pool_bytes']['after']:,} |" for n,name in [(8192,'torus8192-progressive'),(16384,'torus16384-fixed32')]);a,b=g['cases']['before']['median_ms'],g['cases']['after']['median_ms']
hand=read('prepared-new-cell-hand-repeat-v1/comparison.json');assert hand['complete']
hand_table='\n'.join(row('AB pair' if x['pair']==0 else 'BA pair',x) for x in hand['comparisons'])
text=f'''# Shared geometry for new-cell sphere checks

VTM completion now prepares each unproved tetrahedron's sphere geometry once,
then queries the existing current-site segment hierarchy in groups of 32 sites.
It borrows consumed seed geometry arrays and adds no device storage. The original
point predicates, symbolic tie rule, error bits and query-counter definitions
remain unchanged. This applies to the existing private 4,097–16,384-site route;
standalone diagram and Delaunay construction keep their existing paths.

## Complete resident solves

Production `3b68a60` is compared with main `a4f7215` on NVIDIA L40 / Warp 1.19.0.
Seven fresh interleaved pairs each record three complete measured replays,
including domain refit, sampling, fitting, refinement/quality and extraction.
Setup, compilation, transfers and offline validation are excluded. All initial
and measured windows have zero CPU quota throttling. Raw samples and setup costs
are retained in the [comparison](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-final-v2-independent.json).

The first batch stopped when the graded candidate recorded one CPU-quota
throttling period (about 54 ms). Its valid mesh and raw timings are retained as
diagnostic evidence. A fresh graded control/candidate pair passed the same
zero-throttling gate; three earlier clean pairs were retained without repeating
them. The primary cohort therefore contains seven complete clean pairs. The
[affected pair](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-throttled-pair-independent-v1.json)
is independently validated as well and is excluded from the primary timing table.

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
{table}

All fourteen saved meshes pass independent exact full-hull/orientation/empty-sphere,
material topology/quality, boundary vertex-link and boundary self-intersection
checks. Coordinates, canonical full/material cells, oriented boundary, roles and
solver work match in every pair; graded VTU exports round-trip. These are GPU
improvements. The [wide comparable-quality CPU goal](fixed-site-current.md)
remains unmet, and README retains its independently earned Paragram flagship.

## Current-input proof and scratch ownership

The old checker repeated identical Sphere/Box construction in all 32 lanes and
scanned every site. The two [production kernels](../warpower/_unproved_queries.py)
use those same operations once per current unproved cell. Fast math and fused
operations stay disabled. A segment node encloses its descendants: strict box
separation implies the old point-box test would reject every descendant, so
skipped nodes remove no counted queries. Remaining point-box, general Ball64
outside and exact/symbolic checks are unchanged. Query order changes, while
integer additions and error-bit ORs retain their meaning for a given proof mask.
This does not promise identical existing atomic duplicate-winner provenance
across unrelated runs. The [proof note](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-proof-v1.md)
records the finite original-input and current-hierarchy prerequisites.

Neither preparation nor queries stop on a preexisting error bit. Invalid
predicates and nonempty spheres still accumulate bits 2 and 8. Each active cell
increments exactly one skipped/checked proof counter. Proved cells never read
stale geometry; each unproved entry is prepared immediately before use.

[Completion](../warpower/_proved_completion.py) borrows the seed checker's
Sphere/Box arrays only after its queries and compaction finish. Every subsequent
seed or certificate pass refreshes its own current geometry. The helper owns
direct array references, with no parent cycle or global registry. Capacity,
device and current permutation association are checked; insufficient-capacity
or unsupported routes retain the old checker. No proof crosses an input update.

| Reserved sites | Before pool bytes | After pool bytes |
| --- | ---: | ---: |
{pools}

These are persistent workspace pool measurements, not process peak memory.
No device arrays are added. The new preparation launch is included in all timings.

## Why the strict/exact split was not adopted

A second private variant added existing certified strict filters after the
original query-count increment, borrowing consumed seed masks/index storage for
exact fallback. It passed 56 primitive/dispatch and 77 integration tests, including
lane 31, overflow rescans and a later query that rejects a cell after an earlier
query accepts it. All eight saved full meshes and both grid outputs passed
independent checks and matched their controls.

It provided no useful extra speed. In diagnostic fixed 16k profiles the simple
checker took 464.820 ms; the split strict pass took 434.386 ms plus 40.593 ms exact.
Both additionally prepared geometry in about 6.75 ms. The strict saving was smaller
than the added exact work. Clean fixed 16k and graded results likewise favored the
simpler version. Additional simple 8k/grid candidates used retained fresh split
controls, rather than new interleaved pairs; their outputs independently match
both controls and split candidates. The [decision](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-adoption-decision-v1.md)
and [manifest](../results/validation/vtm/prepared-new-cell-queries/manifest.json)
preserve both variants and their exact experimental scope.

The first simple-grid wrapper followed the virtual-environment interpreter
symlink to system Python and failed at NumPy import before GPU setup. That failed
attempt is retained with the successful explicit-venv retry; no meshing failure
was discarded.

## Tests, memory and captured loops

The complete suite passes **1,279 tests**, including 119 focused completion/query
cases. The 42 new cases cover six captured phases with changing coordinates,
counts and proof flags; poisoned scratch; exact error/query/proof-counter
comparison; independent rational small cases; ties, duplicates, flat cells,
scales 2^-60/1/2^60, partial groups, 16K points and 32/256-thread blocks. Capacity
fallback and storage/association guards are included. Existing large-route tests
verify shared-array identity while exercising actual changing-input captures.

All **28 isolated memory jobs** pass with zero memcheck errors, including padded
access checks and pool-disabled lifetime checks. See the [commands and results](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-memory-v1-commands.json).
No timeout or failed memory job is represented as a pass.

Native traces cover a warmed complete 16K solve and a device-controlled
producer/mesher/consumer loop. The latter changes target size and domain,
rejects an invalid count and collapsed input, then restores the original frame.
All three valid loop meshes pass independent checks; rejected frames publish
zero outputs/consumed volume, pointers remain stable, and recovery is exact.
The [full audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-full-native-v1-audit.json)
and [loop audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-loop-native-v1-audit.json)
record {full['gpu_kernel_activities']:,} and {loop['gpu_kernel_activities']:,}
kernel activities. DOT nodes are device kernels, conditional work, memset and
DtoD copies; no host callback, host transfer or explicit allocation/free occurs
inside the measured work. Nsight may omit graph-copy activity records, so every
DOT copy node is also checked. Each trace has one graph launch and final external
synchronization. The first loop launch includes deferred instantiation/upload
before that launch; this is not a claim of zero driver-internal host work.

The warmed full trace totals {profile['total_kernel_milliseconds']:,.3f} ms of kernel
activity. It is diagnostic, not clean performance evidence:

| Kernel | Calls | Registers/thread | Total ms |
| --- | ---: | ---: | ---: |
{kernels}

## Independent API and structured-grid guards

| Callable workload | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
{standalone}

The standalone pairs use nine complete Delaunay replays and thirty diagram runs,
with matching inputs/counts/status/diagnostics and clean timing windows. Their
implementation files are unchanged; small timing differences are not attributed
to this private optimization. Source/kernel AST checks cover 149 source hashes
and verify 141 other runtime/benchmark/example files unchanged.

The primary fitted-hand pair measured a 2.23% increase with a bimodal candidate
sample set. A predeclared ABBA repeat kept every original sample and repeated
both source versions, with the second pair in reverse order:

| Follow-up pair | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
{hand_table}

All follow-up diagnostics match and their timing windows are clean. The
[repeat record](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-hand-repeat-v1/comparison.json)
retains the order and raw samples; the original primary measurement remains in
the table above. This guard uses the unchanged independent Delaunay path.

A [dependency and trace audit](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-hand-trace-v1/nonregression-conclusion-v1.json)
finds 40 byte-identical loaded project modules and identical input fixture bytes;
neither modified runtime module is loaded by this workload. All 18 profiled full
replays have the same 143-kernel configuration signature. The slow/fast behavior
appears in the existing local-sphere checker in both versions (baseline
0.503–0.912 ms, candidate 0.512–0.894 ms). This establishes path independence and
localizes the observed variation; it does not prove a low-level cause such as row
ordering or scheduling. [#235](https://github.com/alecjacobson/warpower/issues/235)
tracks that existing variability. Profiling perturbs timings and does not replace
the clean samples above.

A 17³ retained-grid guard (4,913 sites) measures nine certified replays after
initial construction: **{a:.6f} → {b:.6f} ms**, a {100*(b/a-1):+.2f}% change.
Both outputs independently certify the complete hull and physical quality and
have identical coordinates/canonical cells. This is retained-mesh stress, not a
fresh VTM solve or CPU speedup comparison.

Tracked in [#233](https://github.com/alecjacobson/warpower/issues/233). Larger
active-prefix routing [#230](https://github.com/alecjacobson/warpower/issues/230),
scratch sharing [#122](https://github.com/alecjacobson/warpower/issues/122), and
Armadillo extraction failures [#228](https://github.com/alecjacobson/warpower/issues/228)
remain open. No extraction guard or quality criterion is weakened.

An [additional cross-reservation comparison](../results/validation/vtm/prepared-new-cell-queries/larger-prefix-reservation-comparison-v2.json)
finds identical validated coordinates, connectivity, boundary and solver work for
the 4,096-site target with reservations of 4,096 and 16,384. Current medians are
1,510.678 and 7,220.299 ms (4.78×). This motivates #230; it is a reservation-cost
observation, not a result of a proposed larger-prefix implementation.
'''
p=Path('docs/prepared-new-cell-queries.md');assert not p.exists();p.write_text(text)
for name,anchor,addition in [
 ('docs/vtm.md','The wide comparable-quality CPU performance goal remains unmet.','[Shared new-cell geometry](prepared-new-cell-queries.md) further reduces completion\nwork without adding device arrays.\n'),
 ('docs/delaunay-reuse.md','The measurements in this report describe the earlier implementation.','A later [new-cell checker](prepared-new-cell-queries.md) borrows consumed seed\ngeometry for current-cell preparation and hierarchical queries.\n'),
 ('docs/architecture.md','Proofs are re-established against current input on the next invocation.','On the larger private routes, [new-cell sphere geometry](prepared-new-cell-queries.md)\nis prepared once and queried through the current segment hierarchy. Consumed seed\nSphere/Box arrays supply the scratch, with no extra device storage; the original\npoint predicates and query/error/proof-counter definitions remain unchanged.\n')]:
 p=Path(name);s=p.read_text();assert s.count(anchor)==1
 s=s.replace(anchor,addition+anchor) if name=='docs/vtm.md' else s.replace(anchor,anchor+'\n'+addition.rstrip())
 p.write_text(s)
p=Path('docs/cached-exact-seed-queries.md');s=p.read_text();old='''checks, memory logs and native traces. New-cell queries
[#233](https://github.com/alecjacobson/warpower/issues/233), larger active-prefix
routing''';new='''checks, memory logs and native traces. The later [new-cell checker](prepared-new-cell-queries.md)
shares current geometry and traverses existing bounds. Larger active-prefix
routing''';assert old in s;s=s.replace(old,new);p.write_text(s)
