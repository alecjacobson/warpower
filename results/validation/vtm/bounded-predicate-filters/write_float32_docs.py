"""Write final documentation only from completed validation records."""
from pathlib import Path
import json
root=Path.cwd();w=root/'.work/vtm'
final=json.loads((w/'float32-final-independent.json').read_text())
assert len(final['cases'])==8 and len(final['standalone'])==5
memory=json.loads((w/'float32-memory-commands.json').read_text());assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
native=json.loads((w/'float32-native-audit-summary.json').read_text())
assert '795 passed' in (w/'float32-full-suite.log').read_text()
assert '150 passed' in (w/'float32-focused-tests.log').read_text()
labels={'torus1000':'Uniform torus, 1,000 sites','torus4096':'Uniform torus, 4,096 sites','graded-torus':'Graded torus, 4,096 sites','thin-hand':'Thin hand, 1,000 sites','cavity':'Cavity, 1,000 sites','disconnected':'Disconnected solids, 1,000 sites','armadillo-uniform':'Armadillo uniform, 4,096 sites','armadillo-graded':'Armadillo graded, 4,096 sites'}
rows='\n'.join(f"| {labels[k]} | {r['before_ms']:.3f} | {r['after_ms']:.3f} |" for k,r in final['cases'].items())
standalone='\n'.join(f"| {k} | {r['before_ms']:.6f} | {r['after_ms']:.6f} |" for k,r in final['standalone'].items())
text=f'''# Certified bounded float32 filters

Private VTM facet search and retained-cell queries now try a binary32 strict
predicate before their existing binary64 and exact paths. The new arithmetic
admits only finite original float32 coordinates with absolute value at most 16.
An unsupported range or uncertain sign uses the unchanged fallback. It neither
normalizes inputs nor changes geometric ties, symbolic ordering or final
certification. Public diagram and ordinary Delaunay defaults retain their kernels.

The [error proof](bounded-predicate-proof.md) covers the determinant, its permanent,
operand/result subnormal flushing and the rounded threshold itself. It uses
conservative relative-factor counts and exact rational absolute-error envelopes.
The absolute floor is a rounding allowance, never an approximate tie criterion.
The executable proof checks the current expression trees, range gates, constants
and unfused/non-fast-math consumer options before checking those inequalities.
Only signs are consumed; rounded determinant magnitudes are not geometric outputs.

Facet queries fall back through the existing analytic binary64 filter and original
interval/integer/symbolic policy. Retained cells fall back through the existing
prepared point-specific Ball64 expression and original symbolic predicate.
Group boxes, query order, completion rounds, topology and coverage checks, proof
lifetimes and error publication remain unchanged. All decisions stay on device.
There are no added persistent arrays, graph nodes or kernel launches.

## Complete solves

Runtime `{final['runtime'][:7]}` is compared with baseline `b5d369c` / main
`c5b75f4` on NVIDIA L40 and Warp 1.19.0. Controls are fresh private paired runs;
the integrated runtime is measured again after its full test suite. Every first
and timed quota window is unthrottled. Each replay includes domain refit, fresh
sampling, growth, fitting, quality stages, extraction and diagnostics. Setup,
compilation, upload and offline checking are excluded from replay. The 1k torus
uses five measured replays; the other complete examples use three.

| Complete fresh solve | Before, ms | Bounded filters, ms |
|---|---:|---:|
{rows}

These compare GPU implementations. They do not establish the outstanding
[quality-matched CPU outperformance goal](cpu-size-frontier.md).
The allocator-pool footprint is unchanged in the 1k and 4k torus measurements.
Setup and first-execution measurements
are retained in the raw records; setup is not repeatedly timed, and no cold-start
speedup is claimed. The separate private face-only, seed-only and combined pilot
results are also retained.

All eight final meshes preserve full/material tetrahedron sets, oriented boundaries,
site roles, work counts, input arrays and parameters. The validation JSON records
any coordinate differences explicitly. Six pairs preserve positions bitwise;
the hand differs in three coordinates by at most 2.223e-18 and the cavity differs
in one coordinate by one float32 ULP (2.328e-10). Each full hull passes independent CGAL
exact orientation, global sphere, incidence and hull-support checks. NumPy/AABB
checks evaluate material quality and fit and verify single-cycle boundary vertex
links. All six gallery outputs round-trip through VTU exactly and have zero
CGAL boundary self-intersection pairs.

## Standalone routines

The ordinary independently callable diagram and Delaunay paths are unchanged.
The following fresh comparisons use nine replays per Delaunay operation and
thirty per diagram operation. All measured quota windows are clean. Timing
variation here is not attributed to the VTM filters.

| Operation | Before, ms | After, ms |
|---|---:|---:|
{standalone}

## Correctness and device execution

The integrated focused suite passes **150 tests**; the full suite passes
**795 tests**. Nine new cases check exact rational enclosure/accepted signs,
explicit operand/result-flush simulation and the executable algebraic proof.
Existing captured face and seed-query tests exercise changing coordinates,
canonical owners, permutations, poisoned state and empty/shrinking/restored
prefixes at block sizes 32/256, including scales outside the new range and ties.

The changed arithmetic and both query consumers pass padded Compute Sanitizer
memcheck and API-memory/full allocation-lifetime checking. A routed 2,050-site
integration case passes padded checking and API-memory/use-after-free tracking.
All eight jobs complete with zero reported errors, using a 128 KiB diagnostic
stack and disabling the diagnostic array pool for lifetime checks. Production
stack/pool settings are unchanged. The routed case does not claim a new full
allocation-order pass; the existing
[conditional CUB instrumentation limitation](cuda-validation.md#conditional-native-radix-sort-scratch)
remains separately documented.

NATIVE_AUDIT_DETAILS_PENDING

A counter-instrumented 4k trajectory spans three complete graph executions. It
records 2,293,364,355 facet orientations, of which 7,080 require binary64, and
1,122,325,146 facet sphere queries, of which 9,875,715 require binary64. The
retained-cell path evaluates 1,816,319,529 point-sphere queries, of which
117,866,769 require prepared Ball64. No further fallback is needed on this
trajectory; degenerate tests exercise the unchanged exact/symbolic paths.
Diagnostic positions are bitwise equal, cell sets and work counts match, and
all counters use int64. Counter timings are excluded from performance evidence.

Sources, arithmetic proof output, private variants, initial test-harness compilation
failures, complete meshes, raw timings, memory checks and native traces are retained
in [validation artifacts](../results/validation/vtm/bounded-predicate-filters/manifest.json).
Tracked in [#180](https://github.com/alecjacobson/warpower/issues/180).
'''
(root/'docs/bounded-predicate-filters.md').write_text(text)
