"""Write the result report from completed production evidence, then link it."""
import json
from pathlib import Path
w=Path('.work/vtm')
def read(name):return json.loads((w/name).read_text())
r=read('hierarchical-cache-final-v1-independent.json');g=read('hierarchical-cache-grid-v1-independent.json')
assert r['complete'] and g['complete']
profile=read('hierarchical-cache-full-native-v1-kernels.json');mem=read('hierarchical-cache-memory-v1-commands.json');assert len(mem)==28 and all(x['exit_code']==0 for x in mem)
full=read('hierarchical-cache-full-native-v1-audit.json');loop=read('hierarchical-cache-loop-native-v1-audit.json');assert read('hierarchical-cache-native-loop-independent-v1.json')['complete']
labels={'torus16384-fixed32':'Uniform torus 16,384, fixed 32','torus8192-progressive':'Uniform torus 8,192, progressive','torus8192-fixed32':'Uniform torus 8,192, fixed 32','reserved16384-target4096':'Torus target 4,096, reserved 16,384','graded-torus16384':'Graded torus 16,384, progressive','torus4096':'Uniform torus 4,096','torus1000':'Uniform torus 1,000'}
def row(name,x):return f"| {name} | {x['before_ms']:,.3f} | {x['after_ms']:,.3f} | {100*(x['after_ms']/x['before_ms']-1):+.2f}% |"
table='\n'.join(row(labels[name],r['cases'][name]) for name in labels)
standalone='\n'.join(row(name,x) for name,x in r['standalone'].items())
names=['strict_check','exact_check','retained_cached','check_unproved_spheres','search_unproved_faces','record_face_support']
selected=[dict(x,label=next(n for n in names if x['kernel'].startswith(n+'_'))) for x in profile['kernels'] if any(x['kernel'].startswith(n+'_') for n in names)]
assert len(selected)==6
kernels='\n'.join(f"| {x['label']} | {x['calls']} | {x['registers']} | {x['milliseconds']:,.3f} |" for x in selected)
pools='\n'.join(f"| {n:,} | {r['cases'][name]['persistent_workspace_pool_bytes']['before']:,} | {r['cases'][name]['persistent_workspace_pool_bytes']['after']:,} |" for n,name in [(8192,'torus8192-progressive'),(16384,'torus16384-fixed32')])
a,b=g['cases']['before']['median_ms'],g['cases']['after']['median_ms']
text=f'''# Cached exact fallback for VTM seed queries

At VTM capacities 4,097–16,384, strict sphere tests and exact fallback now run
in separate kernels. Each lane caches its first unresolved point. A second
unresolved point marks overflow and preserves the complete lane rescan. The
existing predicates, symbolic tie policy, hierarchy and final certificate stay
in force. Independently callable diagram/Delaunay functions and smaller VTM
routes retain their existing implementations.

## Complete resident solves

Production `17c6361` is compared with main `7c3dfc7` on NVIDIA L40 / Warp 1.19.0.
Seven fresh interleaved pairs each record three complete measured replays,
including domain refit, sampling, fitting, quality/refinement and extraction.
Setup, compilation, transfers and offline validation are excluded. All initial
and measured windows have zero CPU quota throttling. Raw samples and setup costs
are retained in the [comparison](../results/validation/vtm/cached-exact-seed-queries/hierarchical-cache-final-v1-independent.json).

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
{table}

All fourteen meshes pass independent exact full-hull/orientation/empty-sphere,
material topology/quality, boundary vertex-link and boundary self-intersection
checks. Every pair has identical coordinates, canonical full/material cells,
oriented boundary, roles and work counters. The graded VTU exports round-trip.
These are GPU improvements; the wide comparable-quality native CPU goal remains
open. The [CPU study](fixed-site-current.md) records count/quality and timing
limitations. README retains its previously earned Paragram flagship result.

## Why cache the unresolved point

The first private split-only revision improved the fixed and graded 16K cases
by 16.2% and 9.9%, but regressed a structured-grid retained-mesh test by 7.13%
(4.363 to 4.674 ms). Its exact pass repeated the tree traversal. That negative
result, all private sources and all fourteen private normal-case output checks
are preserved. The earlier [small-capacity experiment](prepared-split-probe.md)
also remains unadopted.

On the 17³ integer grid, diagnostic instrumentation observes 50,341 queried cell
instances across three invocations, all requiring exact fallback. Of 199,236
flagged lanes, the revised cache resolves 197,114 directly and rescans 2,122.
The instrumented output matches the independently checked clean mesh. On the
fixed-16K torus, three graph executions query 67,017,258 cell instances across
681 calls without exact fallback. Instrumentation times are not benchmark samples.

The newly interleaved production grid guard measures **{a:.3f} to {b:.3f} ms
({100*(b/a-1):+.2f}%)** over nine retained-mesh replays. Both 24,034-cell outputs
pass independent exact full-hull and physical quality checks and agree exactly.
Initial full construction is outside this guard's timing: it is neither a fresh
VTM solve nor the standalone public Delaunay benchmark.

## Current-invocation evidence and overflow

The strict pass retains all owner/incidence, hierarchy/point-box, bounded
binary32 and prepared Ball64 checks. Certified positive signs reject the cell;
negative signs continue. A rejected cell still visits its other queries.
Each lane saves its first unresolved point index. Further unresolved points
replace it with -2, which selects the original complete lane traversal in the
exact pass. Thus fixed cache capacity never limits the number of exact checks.
A single saved index invokes the unchanged exact/symbolic predicate directly.
Malformed saved indices fail closed.

The exact kernel follows the strict kernel in stream order, against unchanged
input within that invocation. Every invocation clears all pending masks; every
flagged lane writes its saved index before publishing its bit. Unflagged entries
are never read. No cached proof survives a later input update. All storage is
allocated during setup; replay adds a mask reset and ordered exact-query launch.
Replay has no host decision, readback, allocation or dynamic queue.

At 131,072 candidate cells, the 32-int cache adds 16 MiB and the masks add
0.5 MiB. The exact total is 17,301,504 bytes. Pool measurements are persistent
workspace allocator allocations, not process-wide peak memory:

| Reserved sites | Before pool bytes | After pool bytes |
| --- | ---: | ---: |
{pools}

## Validation and native execution

The complete suite passes **1,237 tests**, including 131 focused query/traversal
and integration cases. New tests cover unsigned lane 31, all-lane overflow,
poisoned storage, zero/restored counts, both block sizes, and a decisive overflow
case: its first uncertain point permits the cell but its second rejects it.
Independent rational oracles and the existing flat checker establish the answers.
Captured producer/consumer tests include input updates, duplicates, invalid boxes,
thin clouds and extreme scales through 16,384 sites. Both promoted query-kernel
ASTs match the measured private cache revision. All 141 other runtime, benchmark
and example source files are byte-identical to the control.

Fourteen selected cases each pass isolated padded and pool-disabled lifetime
memory checks: **28 jobs, zero reported errors**. These include cache overflow
and large-route failure/recovery. The older combined-process sanitizer issue
[#155](https://github.com/alecjacobson/warpower/issues/155) remains separate.

Native traces cover a warmed full 16K solve and one device-controlled five-phase
loop: 4,097 sites, a deformed 8,192-site case, invalid target 8,193, collapsed
domain and recovery. The three valid frames pass independent geometry checks;
invalid frames expose zero counts and consumed volume. Output pointers stay
stable and recovery matches the initial mesh. Native graph nodes contain device
kernels, memsets, device-to-device copies and conditional control, with no host
callbacks, host transfers or explicit allocation/free calls. Each trace has one
external final synchronization. The first loop launch includes deferred graph
instantiation/upload before graph launch; this does not claim zero driver-internal
setup work or memory.

The full trace records {full['gpu_kernel_activities']:,} kernel activities and the
loop {loop['gpu_kernel_activities']:,}. Selected full-trace costs are diagnostic,
not additional clean timing samples:

| Kernel | Calls | Registers/thread | Activity (ms) |
| --- | ---: | ---: | ---: |
{kernels}

## Independent API controls

| Independent operation | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
{standalone}

These fresh pairs use nine complete Delaunay replays and thirty diagram runs,
with matching inputs, output counts/status/diagnostics and clean quota windows.
The public APIs remain independently callable and their code paths are unchanged.

Tracked in [#175](https://github.com/alecjacobson/warpower/issues/175). The
[manifest](../results/validation/vtm/cached-exact-seed-queries/manifest.json)
pins sources, every positive/negative experiment, complete timings, independent
checks, memory logs and native traces. New-cell queries
[#233](https://github.com/alecjacobson/warpower/issues/233), larger active-prefix
routing [#230](https://github.com/alecjacobson/warpower/issues/230), scratch sharing
[#122](https://github.com/alecjacobson/warpower/issues/122), and known Armadillo
extraction failures [#228](https://github.com/alecjacobson/warpower/issues/228)
remain separate work. No extraction guard or quality criterion is weakened.
'''
p=Path('docs/cached-exact-seed-queries.md');assert not p.exists();p.write_text(text)
p=Path('docs/architecture.md');s=p.read_text();old='and preparation launch, while retaining the same point predicates and certificate.';assert old in s;s=s.replace(old,old+'\nOn these larger routes, [strict and exact queries run separately](cached-exact-seed-queries.md).\nEach lane caches one unresolved point from the current invocation; overflow\nretains the complete rescan. Fixed setup storage and device-only dispatch preserve\nthe existing exact policy and standalone API paths.');p.write_text(s)
for name in ['hierarchical-seed-queries.md','delaunay-reuse.md']:
 p=Path('docs')/name;s=p.read_text();at=s.index('\n\n')+2;s=s[:at]+'The subsequent [cached exact-query split](cached-exact-seed-queries.md) builds on\nthis traversal with separate strict/exact kernels and bounded fallback storage.\nThe measurements in this report describe the earlier implementation.\n\n'+s[at:];p.write_text(s)
p=Path('docs/vtm.md');s=p.read_text().replace('The wide comparable-quality CPU performance goal remains unmet.','The [cached exact-query split](cached-exact-seed-queries.md) further improves\nthese larger solves while retaining complete exact fallback.\nThe wide comparable-quality CPU performance goal remains unmet.');p.write_text(s)
