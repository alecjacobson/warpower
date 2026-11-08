"""Write the adoption report only after production validation has completed."""
import json,subprocess
from pathlib import Path
w=Path('.work/vtm');out=Path('results/validation/vtm/hierarchical-seed-queries')
r=json.loads((w/'large-stackless-final-v1-independent.json').read_text());assert r['complete']
assert all(x['comparison']['points_bitwise_equal'] for x in r['cases'].values())
mem=json.loads((w/'large-stackless-memory-v1-commands.json').read_text());assert len(mem)==24 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in mem)
assert '1227 passed' in (w/'large-stackless-full-suite-v1.log').read_text()
assert json.loads((w/'large-stackless-native-loop-independent-v1.json').read_text())['complete']
names={'torus16384-fixed32':'Uniform torus 16,384, fixed 32','torus8192-progressive':'Uniform torus 8,192, progressive','reserved16384-target4096':'Torus target 4,096, reserved 16,384','graded-torus16384':'Graded torus 16,384, progressive','torus4096':'Uniform torus 4,096, progressive','torus1000':'Uniform torus 1,000, progressive'}
rows=[]
for name,d in r['cases'].items():rows.append(f"| {names[name]} | {d['before_ms']:,.3f} | {d['after_ms']:,.3f} | {100*(d['after_ms']/d['before_ms']-1):+.2f}% |")
standalone=[f"| {name} | {d['before_ms']:.3f} | {d['after_ms']:.3f} | {100*d['relative_change']:+.2f}% |" for name,d in r['standalone'].items()]
source=json.loads((w/'large-stackless-full-suite-v1.json').read_text())['source']
s=f'''# Hierarchical checking of retained VTM cells

For VTM capacities 4,097–16,384, the retained-cell filter now traverses the
existing current segment hierarchy in groups of 32 sites. It can exclude a
whole subtree before inspecting any of its groups. The point predicates,
symbolic tie policy and final complete-mesh certificate are unchanged. The
independently callable power-diagram and Delaunay paths, and VTM capacities
through 4,096, keep their existing query implementations.

## Complete resident solves

Production `{source[:7]}` is compared with main `5998b13` on NVIDIA L40 / Warp
1.19.0. All six pairs are newly interleaved, with three measured fresh complete
replays each. The scope includes domain refit, sampling, fitting, refinement,
quality stages and extraction. Compilation, setup, transfers and independent
validation are excluded. All initial and measured timing windows have zero CPU
quota throttling. Raw samples, setup and persistent pool measurements are retained.

| Sites and schedule | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

All twelve outputs pass independent exact full-hull/orientation/empty-sphere
checks, material quality/topology, single-cycle boundary vertex links, and
boundary self-intersection checks. Every pair has bitwise-identical coordinates,
matching canonical full/material tetrahedra, oriented boundary faces, roles and
recorded work. Graded-gallery VTU exports round-trip exactly. See the
[complete comparison](../results/validation/vtm/hierarchical-seed-queries/large-stackless-final-v1-independent.json).

The earlier private six-pair pilot likewise preserved all coordinates,
connectivity and work. It improved the 16k fixed-site solve by 12.1% and the graded
16k solve by 8.2%. Its setup-only registry is absent from production: the existing
hierarchy is passed explicitly when the workspace is constructed. Traversal
kernels have identical ASTs to that measured prototype.

## Coverage and conservative bounds

Let L be the existing segment tree's power-of-two leaf base and G=max(1,L/32).
At depth log2(G), nodes G through 2G−1 cover consecutive blocks of 32 Morton
ranks. Lanes 0–31 inspect those ranks, masking padding beyond the point count.
The existing bottom-up min/max bounds enclose every current descendant point.
The caller rebuilds this segment and the shared permutation before querying.

A depth-first traversal descends to the left child or skips a completed subtree
by climbing right-child ancestors and selecting the next sibling. Thus it visits
each required group exactly once, in increasing rank order, without a per-lane
stack. At most 2G−1 nodes can be visited. An unfinished traversal at that budget
fails closed with an error and clears the cell's live flag.

Only the existing conservative sphere-box exclusion skips a node. An unresolved
or invalid sphere box forces traversal. Each surviving lane keeps exactly the
existing duplicate-owner, incident-site, point-box, bounded float32, prepared
Ball64 and exact/symbolic checks. A rejected cell continues checking its other
queries, preserving diagnostics. No proof survives a change of input.

The production path borrows arrays already owned and rebuilt by the site
hierarchy. It removes the separate group-box arrays and their preparation launch
on the selected routes; it adds no query stack, hierarchy build, host readback,
allocation or device control decision during replay. This is a traversal change,
not a new arithmetic filter or an approximate Delaunay test.

## Independent API performance and validation

| Independent operation | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(standalone)+'''

These newly interleaved measurements cover nine full Delaunay replays and thirty
power-diagram runs. Inputs, counts, status and diagnostics match, with clean quota
windows. All other 139 existing runtime/benchmark/example files are byte-identical
to the control; the two modified existing runtime modules change setup selection
only. Public defaults and README remain unchanged.

The complete suite passes **1,227 tests**, including 106 new primitive cases.
Recursive CPU coverage oracles test partial groups, root pruning, mixed pruning,
empty/full traversal and repeated captured launches. Independent containment
checks cover extreme finite/subnormal coordinates and current permutations.
Captured query comparisons cover ties, duplicates, thin and random clouds,
synthetic torus samples through 16,384 sites, scales 2^-60/1/2^60, empty prefixes,
changing positions and poisoned sphere boxes. Small cases also use exact rational
physical/symbolic oracles. Fifteen existing integration cases exercise route
selection, captured producers/consumers, invalid inputs, fallback, exhaustion and
recovery. The private pilot additionally checks saved real meshing outputs.

Twelve selected primitive/integration cases each pass padded and pool-disabled
lifetime memory checks in isolated processes: **24 jobs, zero reported errors**.
The earlier combined-process sanitizer issue remains tracked separately in
[#155](https://github.com/alecjacobson/warpower/issues/155).

Native audits cover a warmed full 16k solve and a single device-controlled
five-phase loop: 4,097 sites, a deformed 8,192-site case, invalid target 8,193,
a collapsed domain, and recovery. Invalid phases expose zero counts and zero
consumed volume. The three valid snapshots pass independent full/material/boundary
checks; first and recovered outputs agree. Buffers remain stable. Native nodes
contain kernels, memsets, device-to-device copies and conditional control, with
no host callbacks, host copies or explicit allocation/free calls. Each profile
has one final external synchronization. The first loop launch includes deferred
executable-graph instantiation/upload before its single graph launch; initial
setup is not claimed to require zero host work or driver storage. Native timings
are diagnostic, not additional clean benchmark samples.

## Remaining work

This improves the GPU implementation. The wide comparable-quality native CPU
speedup goal remains open; counts, quality and timing scopes must be matched
before such a claim. The [current CPU study](fixed-site-current.md) records those
limitations. Unchanged Armadillo extraction failures remain in
[#228](https://github.com/alecjacobson/warpower/issues/228); no extraction guard is
weakened. Larger active-prefix routing and scratch sharing remain separate
[#230](https://github.com/alecjacobson/warpower/issues/230) and
[#122](https://github.com/alecjacobson/warpower/issues/122) work.

The [provenance manifest](../results/validation/vtm/hierarchical-seed-queries/manifest.json)
links sources, command lines, complete samples, independent checks, native traces
and memory logs. Tracked in [#177](https://github.com/alecjacobson/warpower/issues/177).
'''
p=Path('docs/hierarchical-seed-queries.md');assert not p.exists();p.write_text(s)
p=Path('docs/architecture.md');s=p.read_text();needle='one preparation launch while keeping standalone construction paths unchanged.';assert s.count(needle)==1
s=s.replace(needle,needle+'\nAt capacities 4,097–16,384, the VTM filter instead [traverses the existing\ncurrent segment hierarchy](hierarchical-seed-queries.md) down to those 32-site\ngroups. This skips excluded subtrees and removes the separate group-box buffers\nand preparation launch, while retaining the same point predicates and certificate.')
p.write_text(s)
p=Path('docs/delaunay-reuse.md');s=p.read_text();needle='the exhaustive seed filter costs O(sites × retained cells). The later report\nrecords complete uniform/graded torus and Armadillo gains, storage costs and\nvalidation of the extension. The measurements below describe the earlier change.';assert needle in s
s=s.replace(needle,'the exhaustive worst case remains O(sites × retained cells). The newer\n[hierarchical traversal](hierarchical-seed-queries.md) prunes current segment\nsubtrees on capacities 4,097–16,384. The extension report records complete\nuniform/graded torus gains, unchanged Armadillo failures, storage costs and\nvalidation. The measurements below describe the earlier change.')
p.write_text(s)
p=Path('docs/vtm.md');s=p.read_text();needle='The wide comparable-quality CPU performance goal remains unmet.';assert s.count(needle)==1
s=s.replace(needle,'The [hierarchical retained-cell queries](hierarchical-seed-queries.md) improve\nlarge complete GPU solves while preserving their coordinates and connectivity.\n'+needle)
p.write_text(s)
p=Path('docs/extended-fragment-reuse.md');s=p.read_text();needle='tracks hierarchical traversal. Larger active prefixes still pay for the full';assert needle in s
s=s.replace(needle,'is now implemented by the separately measured [hierarchical seed queries](hierarchical-seed-queries.md).\nThe timings in this report predate that follow-up. Larger active prefixes still pay for the full')
p.write_text(s)
native=json.loads((w/'large-stackless-full-native-v1-kernels.json').read_text())
profile='## Native cost and persistent storage\n\nA separate warmed full 16k fixed-site trace records the following kernel\nactivity. These diagnostic times are not clean complete-solve samples.\n\n| Kernel family | Calls | Activity (ms) |\n| --- | ---: | ---: |\n'
for x in native['kernels'][:5]:
 name=x['kernel'].rsplit('_',4)[0]
 profile+=f"| {name} | {x['calls']} | {x['milliseconds']:,.3f} |\n"
profile+='\nThe query kernel still uses 255 registers per thread. This observation alone\ndoes not identify a performance cause.\n\n| Reserved sites | Before pool bytes | After pool bytes | Saved bytes |\n| --- | ---: | ---: | ---: |\n'
for name in ['torus8192-progressive','torus16384-fixed32']:
 d=r['cases'][name]['persistent_workspace_pool_bytes'];n=8192 if '8192' in name else 16384
 profile+=f"| {n:,} | {d['before']:,} | {d['after']:,} | {d['before']-d['after']:,} |\n"
profile+='\nThese are persistent workspace allocator-pool allocations, not process-wide\npeaks. Small-route storage is unchanged.\n\n'
p=Path('docs/hierarchical-seed-queries.md');s=p.read_text();needle='## Independent API performance and validation';assert needle in s;s=s.replace(needle,profile+needle);p.write_text(s)
