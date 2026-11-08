"""Write the final measured extension report after every required check passes."""
import json
from pathlib import Path
w=Path('.work/vtm');report=json.loads((w/'extended-reuse-final-v1-independent.json').read_text());assert report['complete']
loop=json.loads((w/'extended-reuse-native-loop-independent-v1.json').read_text());assert loop['complete']
assert '1121 passed' in (w/'extended-reuse-full-suite-v1.log').read_text()
qualification=json.loads((w/'extended-reuse-qualification-v2.json').read_text())
assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
memory=json.loads((w/'extended-reuse-memory-v2-commands.json').read_text())
assert len(memory)==30 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
rows=report['cases'];artifact='../results/validation/vtm/extended-fragment-reuse/'
text='''# Certified VTM fragment reuse through 16,384 sites

The active VTM reconstruction route now uses the existing certified fragment and
fresh-wall strategies for reserved capacities from 2,049 through 16,384 sites.
The implementation changes one setup bound. The original-coordinate proof
kernels, ordinary fallback constructor, independently callable power-diagram and
Delaunay paths, and meshing schedules are unchanged.

This extends the earlier [4,096-site integration](large-fragment-reuse.md).
Every reconstruction checks retained cells against the current original input.
Those proofs survive completion within that reconstruction; no proof survives an
input change. Fresh solves clear retained fragments. Failed fragment/wall
completion still uses the ordinary certified constructor, and finite output
exhaustion still returns an error with zero counts. Reserved capacities above
16,384 retain the previous full-construction strategy.

## Complete resident solves

Production `f8e2834` is compared with control `689fd31` on NVIDIA L40 / Warp 1.19.0.
The four ordinary large-torus controls were recorded at `b44ab9e`, whose runtime
is identical to `689fd31`. The graded control comes from the preceding private
study; the small/reserved and standalone pairs are newly interleaved runs.
Three measured complete fresh replays include refit, sampling, fitting,
refinement, quality stages and extraction. Setup, compilation, transfers and
offline checks are excluded. All reported successful timing windows have zero
CPU quota throttling. Separate setup/initial execution and all raw samples are
retained.

| Sites and schedule | Control (ms) | Production (ms) | Ratio |
| --- | ---: | ---: | ---: |
'''
for name,label in [('torus8192-progressive','Uniform torus 8,192, progressive'),('torus8192-fixed32','Uniform torus 8,192, fixed 32'),('torus16384-progressive','Uniform torus 16,384, progressive'),('torus16384-fixed32','Uniform torus 16,384, fixed 32'),('graded-torus16384','Graded torus 16,384, progressive'),('reserved16384-target4096','Torus target 4,096, reserved 16,384')]:
    r=rows[name];text+=f"| {label} | {r['before_ms']:,.3f} | {r['after_ms']:,.3f} | {r['speedup']:.2f}× |\n"
valid=[r for r in rows.values() if r['before_valid'] and r['after_valid']]
bitwise=sum(r['comparison']['points_bitwise_equal'] for r in valid)
text+=f'''
All {len(valid)} successful control/production pairs preserve canonical full/material
connectivity, oriented boundaries, roles and recorded iteration work; {bitwise} have
bitwise identical positions. Each new successful output passes independent CGAL
exact full-hull/orientation/empty-sphere checks, material topology and quality
measurements, single-cycle boundary vertex links, and a boundary self-intersection
check with zero pairs. Retained earlier controls have the same independent checks.
Gallery VTU exports round-trip exactly. See the [complete comparison record]({artifact}extended-reuse-final-v1-independent.json).

The five earlier private uniform/graded torus pairs also preserve bitwise
positions and canonical connectivity, with 5.77–8.43× gains. The production change
has exactly the private prototype's substituted class source; all other runtime
files and predicate kernels are unchanged.

## Small cases and independent APIs

| Complete torus solve | Control (ms) | Production (ms) |
| --- | ---: | ---: |
'''
for name,label in [('torus1000','Target/capacity 1,000'),('torus4096','Target/capacity 4,096'),('reserved8192-target1000','Target 1,000, capacity 8,192')]:
    r=rows[name];text+=f"| {label} | {r['before_ms']:,.3f} | {r['after_ms']:,.3f} |\n"
text+='''
Standalone timings cover the entire existing operation, with nine Delaunay
replays and thirty power-diagram runs. Their source paths are unchanged.

| Independent operation | Control (ms) | Production (ms) | Change |
| --- | ---: | ---: | ---: |
'''
for name,r in report['standalone'].items():text+=f"| {name} | {r['before_ms']:,.3f} | {r['after_ms']:,.3f} | {100*r['relative_change']:+.2f}% |\n"
text+='''
## Persistent storage and capture

| Reserved sites, progressive torus | Control pool bytes | Production pool bytes | Additional bytes |
| --- | ---: | ---: | ---: |
'''
for name,n in [('torus8192-progressive',8192),('torus16384-progressive',16384)]:
    r=rows[name]['persistent_workspace_pool_bytes'];text+=f"| {n:,} | {r['before']:,} | {r['after']:,} | {r['after']-r['before']:,} |\n"
text+='''
These measure persistent workspace allocator-pool allocations, not process-wide
peaks. A reservation pays the extra storage even when a smaller active target
only uses child routes. No allocations occur during replay.

The full suite passes **1,121 tests**. The 15 expanded large-route cases cover
capacities 2,050/4,096/4,097/8,192/16,384; active prefixes across 1,024/2,048;
captured producer/consumer work; nonfinite input and recovery; fresh resets;
corruption/fallback and output exhaustion at 2,050/8,192; and cospherical duplicate
ownership at scales 2^-60, 1 and 2^60. All 15 pass both padded memory checking and
pool-disabled allocation-lifetime checking with zero reported errors, in fresh
processes for each parameterized case. The first combined sanitizer attempt
stalled at setup synchronization after two test markers; its interrupted log and
failed qualification record are retained, not counted as a pass. The unresolved
combined-process behavior is tracked in
[#155](https://github.com/alecjacobson/warpower/issues/155).

A native 16k fixed-site full solve and a single device-controlled five-phase loop
show kernel, memset, device-to-device copy and conditional nodes only. There are
no host callbacks, host copies or allocation/free nodes/APIs. The loop reserves
8,192 sites and exercises 4,097 sites, deformed geometry at 8,192, an invalid
8,193 target, a collapsed domain, and recovery to 4,097. Both invalid phases expose
zero counts and zero downstream consumed volume; all three successful snapshots
pass independent hull/material/boundary checks. Output pointers stay fixed.
Each profile contains one final external synchronization. Explicit array/workspace
setup and final readback are outside the trace. The first loop launch also records
Warp’s deferred executable-graph instantiation and upload before its one graph
launch; these are initial preparation, not per-phase host decisions. The warmed
full-solve trace has no such calls. Driver-internal preparation is not claimed to
require zero host work or internal storage. Profiling durations are diagnostic,
not clean benchmark timings.

## Remaining work

Uniform and graded Armadillo at 8,192 sites fail extraction in both control and
extension. The graded case also fails with doubled surface quadrature. These
failed outputs and timings are retained without a successful-mesh speedup claim;
one denser control has CPU quota throttling. In the graded diagnostic, removing
six exterior-center cells repairs four bad edges but leaves two disconnected
boundary fans at vertex 6,598. The existing guard correctly rolls back and
publishes zero counts. [Issue #228](https://github.com/alecjacobson/warpower/issues/228)
tracks the recovery problem. This PR changes no extraction guard.

These are gains over the GPU implementation. The 16k fixed-site solve remains
slower than the separately measured native parallel CPU result of about 2.44 s,
which has differing counts/quality and includes throttled samples. The
[comparable-quality CPU speed goal](fixed-site-current.md) remains open.

The earlier native diagnostic attributes 3.03 of 7.08 seconds of kernel activity
to grouped seed checks. [Issue #177](https://github.com/alecjacobson/warpower/issues/177)
tracks hierarchical traversal. Larger active prefixes still pay for the full
reservation after 2,048 sites; [#230](https://github.com/alecjacobson/warpower/issues/230)
tracks additional routes, with scratch sharing in
[#122](https://github.com/alecjacobson/warpower/issues/122). These are separate
follow-ups so their timing and storage costs can be measured independently.

Reproduction commands, pinned source and native binary hashes, raw results,
failed cases, and lossless native traces are linked in the
'''+f'[provenance manifest]({artifact}manifest.json).\n'
p=Path('docs/extended-fragment-reuse.md');assert not p.exists();p.write_text(text)
p=Path('docs/architecture.md');s=p.read_text();old='''capacities of 2,049–4,096 sites, that full route now
[uses certified fragment reuse](large-fragment-reuse.md), retaining current-input
cell proofs throughout completion. Larger capacities retain ordinary full
construction.''';new='''capacities of 2,049–16,384 sites, that full route now
[uses certified fragment reuse](extended-fragment-reuse.md), retaining current-input
cell proofs throughout completion. Larger capacities retain ordinary full
construction.''';assert old in s;p.write_text(s.replace(old,new))
p=Path('docs/delaunay-reuse.md');s=p.read_text();s=s.replace('2,049–4,096 sites](large-fragment-reuse.md)','2,049–16,384 sites](extended-fragment-reuse.md)');p.write_text(s)
p=Path('docs/vtm.md');s=p.read_text();old='''until the sites outgrow them. Full original-coordinate certification and the
ordinary full-capacity fallback remain mandatory.''';new='''until the sites outgrow them. Their [full route reuses certified fragments through
16,384 reserved sites](extended-fragment-reuse.md). Full original-coordinate
certification and the ordinary full-capacity fallback remain mandatory.''';assert old in s;p.write_text(s.replace(old,new))
p=Path('docs/large-fragment-reuse.md');s=p.read_text();heading='# Certified fragment reuse through 4,096 sites\n';assert s.startswith(heading);p.write_text(s.replace(heading,heading+'\nThis report describes the earlier 4k integration. The [current full route extends\nthrough 16,384 reserved sites](extended-fragment-reuse.md).\n',1).replace('The active VTM mesher now uses certified fragment reuse on its full reconstruction\nroute','The initial integration enabled certified fragment reuse on the active VTM full\nreconstruction route',1))

p=Path('docs/active-reconstruction.md');s=p.read_text();s=s.replace('[later full-route extension](large-fragment-reuse.md)', '[later full-route extension](extended-fragment-reuse.md)');p.write_text(s)
