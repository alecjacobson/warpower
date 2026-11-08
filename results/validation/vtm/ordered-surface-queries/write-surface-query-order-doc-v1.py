"""Write the ordered-query report from completed immutable measurements."""
import json
from pathlib import Path

root = Path.cwd()
w = root/'.work/vtm'
r = json.loads((w/'surface-query-order-final-v1-independent.json').read_text())
assert r['complete']
names = {'torus16384-fixed32':'Torus, 16,384, fixed / 32 polish', 'torus1000':'Torus, 1,000', 'torus8192-fixed32':'Torus, 8,192, fixed / 32 polish', 'torus8192-progressive':'Torus, 8,192, progressive', 'thin-hand-seed7':'Thin hand, 1,000, seed 7', 'cavity':'Cavity', 'disconnected':'Disconnected', 'plate':'Thin plate', 'armadillo-graded4096':'Graded Armadillo, 4,096', 'armadillo-graded8192-recovery':'Graded Armadillo, 8,192', 'armadillo-uniform8192-recovery':'Uniform Armadillo, 8,192'}
table = '| Complete solve | Before, ms | After, ms | Change | Added pool bytes | Setup before → after, s |\n|---|---:|---:|---:|---:|---:|\n'
for name,c in r['cases'].items():
    t=c['timing']
    table += f"| {names[name]} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {100*(t['after_ms']/t['before_ms']-1):+.2f}% | {t['additional_pool_bytes']:,} | {t['before_setup_seconds']:.3f} → {t['after_setup_seconds']:.3f} |\n"
standalone = '| Standalone case | Before, ms | After, ms | Change |\n|---|---:|---:|---:|\n'
for name,t in r['standalone'].items():
    standalone += f"| {name} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {100*t['relative_change']:+.2f}% |\n"
text = '''# Ordered surface queries

Large surface-query batches in the full mesher now run in spatial order.
The complete 16K torus solve takes 4.409 seconds versus 4.899 seconds on the
NVIDIA L40: 10.0% less replay time, or 1.111× throughput. Two 8K torus schedules
improve by about 8.4%; the graded 8K Armadillo improves by 5.7%.
These compare GPU revisions, not CPU meshers. The wide comparable-quality CPU
speedup goal remains open, and the README flagship is unchanged.

## Implementation

The private `OrderedSurfaceQueryCache` replaces the cache constructor used by
extraction, selective optimization and approximate quality moves. Each cache
with tetrahedron capacity greater than 32,768 owns centroid keys and a cell-index
permutation. A device predicate selects sorting when the current query count is
at least 32,768 and the upstream error is zero. Smaller caches retain the original
scalar launch with no ordering arrays or decision nodes.

The permutation uses 21-bit-per-axis Morton keys in the current refitted domain
box. It changes execution order only: the three classifier kernels preserve
their original arithmetic, predicates, cache lookups and output cell indices.
Invalid/inactive key slots have a sentinel key; their IDs still form a complete
permutation. The original scalar cache implementation is unchanged.

Warp's radix sort runs inside the selected conditional graph. Setup initializes
the sort outside capture and retains the temporary storage needed by each child
graph. Count and error decisions stay on device during replay. The private eager
launch on a large cache always orders; the full mesher uses captured launches.
This adds no custom C++ or CUDA implementation to the mesher.

Ordering payload per eligible cache is `24*capacity + 4` bytes, plus retained sort
scratch. Measured pool increases across the complete workspace are below. The
independently callable diagram and Delaunay functions do not construct these
private surface-query caches; their source and allocations are unchanged.

## Complete solves and standalone controls

Baseline `6d362e7b4477`, runtime `520a68cf5f8d`, Warp 1.19.0, NVIDIA L40.
Each complete solve uses three warmed fresh captured replays, including surface
refit, sampling, growth, fitting, quality moves and extraction. Setup, upload,
readback and export are outside replay timing. All 32 timing jobs have clean
CPU-quota windows. The two 8K torus schedules are additional complete-solve
holdouts; exact controls and all samples are retained in the
[commands](../results/validation/vtm/ordered-surface-queries/surface-query-order-final-v1/commands.json).

'''+table+'''
Setup measurements are single paired observations, not a repeated setup study.
In particular, graded 4K setup rises from 4.231 to 4.851 seconds even though it
allocates no ordering arrays. This measurement is retained without attributing
a cause; [#246](https://github.com/alecjacobson/warpower/issues/246) tracks repeated
setup measurements and investigation. The added workspace does not imply a replay gain on every large case:
uniform 8K is essentially unchanged.

'''+standalone+'''
These short pairs show no material standalone regression in the measured cases;
they are not a statistical guarantee over all workloads.

## Geometry and graph qualification

All 1,346 tests pass. The 30 focused cases include 19 new cases: live device gates
at capacities 64/32,768/32,769, nondefault streams, invalid and changing counts,
upstream failures, collapsed/restored domains, stable addresses, exact scalar
outputs, forced ordered classifiers at scales 1e-30/1/1e30, and inherited cache
refit/cutoff/failure contracts. The inherited direct-query contract exercises
cache behavior; its phases are not all ordered classifier launches.

All 22 final meshes pass the independent complete-hull/Delaunay, positive-cell,
material topology, boundary vertex-link and exact self-intersection checks.
Eighteen receive fresh independent validation; four torus proofs transfer only
through exact arrays, oriented connectivity and source/report hashes. All 14
gallery exports receive fresh VTU round trips. Full and material connectivity,
oriented boundary, roles, inputs and scheduler work agree in every pair.

Nine pairs have bitwise equal coordinates. The hand and cavity each differ in one
scalar coordinate, by at most 3.47e-18 and 2.33e-10 respectively. Both use the small
original scalar path. Their strict trajectory flags remain false; no cause is
claimed. Minimum radius ratio is unchanged within 1e-12 in all pairs. The complete
[geometry comparison](../results/validation/vtm/ordered-surface-queries/surface-query-order-final-v1-independent.json)
retains these differences and the independent quality and sampled fitting metrics.

Native traces cover a full uniform 8K solve and the unchanged five-phase device
producer → mesher → consumer loop. There is one outer graph launch and one final
synchronization, no host copies or callbacks, and no allocation/free API calls in
the replay windows. The loop's first graph instantiate/upload is explicitly
present. All graph copy nodes are device-to-device. Executed kernel activities
increase to 93,800 and 158,566 respectively; ordering adds graph work, so unchanged
node counts are not claimed.

The loop covers exhausted recovery, success, an invalid live target, a collapsed
domain and restored success. It retains stable addresses, resets diagnostics,
zeros public outputs on failure and reproduces the certified oriented geometry
for the exhausted and successful frames. Its ordinary, native, padded-memory
and lifetime-memory outputs all match those references. The 19 new tests and
the full loop each pass padded and pool-disabled lifetime checks: four completed
memcheck jobs, zero errors. Native/sanitizer durations are diagnostic, not clean
performance samples. Earlier combined-process instrumentation limits in
[#155](https://github.com/alecjacobson/warpower/issues/155) remain separate.

## Rejected trials and evidence

The first private implementation failed with CUDA error 900 because a nested sort
graph was lazily instantiated during capture. Its nine failed tests and one pass
are retained. Initializing the sort before capture and capturing its calls
directly fixed that failure. Static cold-batch gains did not predict all full
solves: small and warm-hit batches regressed, and the first 16,384-count gate
slowed the graded 4K complete solve from 2.578 to 2.824 seconds. That variant was
rejected. The final 32,768 gate and small-cache storage omission passed a second
private full-solve trial before production qualification.

The [manifest](../results/validation/vtm/ordered-surface-queries/manifest.json)
pins raw successful and negative trials, meshes, immutable scripts, source hashes,
lossless Nsight/DOT archives and shared reference proofs. A 264-file source audit
finds 259 unchanged files: only the new ordered module, three constructor imports
and its test file differ. The ordered kernel bodies match the original scalar
arithmetic after removing the index permutation. Offline CGAL validators are
outside the mesher and all timing windows.

The Armadillo remains the attributed Stanford/libigl scan, not an assertion that
it is the paper's original model. Its provenance and terms are linked through the
manifest. Existing finite recovery limits and sampled surface-fit limits remain.
This closes the bounded adoption milestone
[#245](https://github.com/alecjacobson/warpower/issues/245);
the broader surface-query work in
[#197](https://github.com/alecjacobson/warpower/issues/197) remains open.
'''
dest=root/'docs/ordered-surface-queries.md'
assert not dest.exists()
dest.write_text(text)
for name, marker in [('architecture.md', "candidate extractor's active cache, ordinary quality-move caches and checkpoints.\n"), ('mesher.md', 'and the candidate extractor retain their active caches.\n'), ('vtm.md', 'saving 12.85 MB at 8K sites without changing kernels or replay methods.\n')]:
    p=root/'docs'/name
    text=p.read_text()
    assert text.count(marker)==1
    text=text.replace(marker,marker+'[Spatial ordering of large surface-query batches](ordered-surface-queries.md)\nreduces measured full-solve replay time by 10.0% on the 16K torus and 5.7% on\nthe graded 8K scan, with unchanged standalone geometry paths.\n')
    p.write_text(text)
print(dest)
