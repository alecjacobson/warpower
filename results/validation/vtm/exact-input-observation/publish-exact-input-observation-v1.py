"""Retain a negative exact-equality opportunity study without changing runtime."""
import gzip, hashlib, json, os, subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/exact-input-observation'
assert not out.exists()
runtime='c2843f658e49c45aff87973c9f70ccaea9dddc88'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((w/n).read_text())
study=read('unchanged-input-study-v3-independent.json');assert study['complete'] and len(study['cases'])==7
assert '1 passed' in (w/'unchanged-input-observer-tests-v1.log').read_text()
assert len(read('unchanged-input-study-v1/commands.json'))==8
assert read('unchanged-input-fixed16k-native-v1-command.json')['exit_code']==0
at=read('unchanged-input-fixed16k-native-v1-attribution.json');assert at['unchanged_calls']==3 and len(at['calls'])==227
native=read('unchanged-input-fixed16k-native-v1.json');assert all(native['work_equal'].values()) and all(native['canonical_cells_equal'].values())
assert native['array_bitwise_equal']['points'] and native['array_bitwise_equal']['roles']
audit=read('unchanged-input-fixed16k-native-v1-audit.json')
assert set(audit['dot_node_types']) <= {'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
assert set(audit['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
assert audit['cuda_api_calls']['cudaGraphLaunch_v10000']==audit['cuda_api_calls']['cuCtxSynchronize']==1
assert all(not any(x in k.lower() for x in ['alloc','free','memcpy']) for k in audit['cuda_api_calls'])
kernel=read('unchanged-input-fixed16k-native-v1-kernels.json')
packed=w/'unchanged-input-fixed16k-native-v1.nsys-rep.gz'
assert sha(packed)==kernel['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==kernel['trace_raw_sha256']
assert hashlib.sha256(gzip.decompress((w/'unchanged-input-fixed16k-native-v1.dot.gz').read_bytes())).hexdigest()==audit['dot_sha256']
for r in read('unchanged-input-study-v1/lossless-archives.json'):
 p=Path(r['path']);assert sha(p)==r['sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==r['raw_sha256']
prefixes=('unchanged-input','unchanged_input','run-unchanged-input','check-unchanged-input','test_unchanged_input','profile-unchanged-input')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-comment-','-pr-','-premerge-','-merge-audit-'])]
files += [p for p in (w/'unchanged-input-study-v1').rglob('*') if p.is_file()]
files += [Path(__file__).resolve()]
meta=dict(scope=__doc__,runtime=runtime,files=[],shared_references=[],runtime_source_sha256={},limitations=['No cache or runtime change adopted; no measured optimization speedup.','Native timeline is instrumented; the 0.42% avoided-span estimate is specific to one fixed16K trajectory.','Six observer cases exactly match independently validated current controls; thin-hand differs at one coordinate by 7.995991022080595e-19 and both outputs validate independently.','v1/v2 checker failures and partial results retained; corrected v3 checks exact hit decisions, geometry/work and independent hand validity.','Broad moved-subset certificate reuse #130 and the wide CPU performance goal remain open.'])
for name in subprocess.check_output(['git','ls-files','warpower','tests','benchmarks','examples','README.md'],text=True).splitlines():
 p=root/name
 if p.is_file():meta['runtime_source_sha256'][name]=sha(p)
for p in [root/'results/validation/vtm/larger-active-prefixes/manifest.json',root/'results/validation/vtm/larger-active-prefixes/larger-prefix-final-v1-independent.json',root/'results/validation/vtm/larger-active-prefixes/profile-larger-prefix-production-v1.py',root/'results/validation/vtm/surface-bound-reuse/check-surface-bound-reuse-final-v2.py',root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py',w/'check-delaunay',w/'surface-distance',w/'surface-reference']:
 meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
for p in sorted(set(files)):
 q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
 row=dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
 os.link(p,q);q.chmod(0o444);assert sha(q)==row['sha256'];meta['files'].append(row)
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
rows='\n'.join(f"| {name} | {r['calls_per_solve']} | {r['unchanged_per_solve']} | {100*r['unchanged_per_solve']/r['calls_per_solve']:.2f}% |" for name,r in study['cases'].items())
doc=f'''# Exact-input reconstruction opportunities

An observation-only study finds too little opportunity to justify an all-input
equality cache for the large fixed-site trajectory. No cache or runtime change
is adopted. This is a bounded result under [#130](https://github.com/alecjacobson/warpower/issues/130);
it does not settle reuse when only a subset of sites moves.

On current runtime `c2843f6`, NVIDIA L40 / Warp 1.19.0, each private constructor
compares all original float32 coordinate bits with its internally saved input
from the last successful construction. Signed zeros remain distinct. Invalid
construction and fresh-solve reset invalidate the checkpoint. Every original
construction and certificate still executes; no result is accepted using the
observer. All routes are observed, including inactive reserved constructors.

| Complete trajectory | Reconstructions per solve | Exactly unchanged | Fraction of calls |
| --- | ---: | ---: | ---: |
{rows}

Each case runs three fresh solves. Exact equality decisions and geometry/error
counts repeat across all three. Small changed-site-count differences occur during
the thin-hand trajectory and remain in the raw records. Counting calls alone
does not measure time saved.

## Native cost attribution

One warmed fixed 16K solve records {at['kernel_activities']:,} kernel activities
over a {at['total_gpu_timeline_span_ms']:,.3f} ms GPU timeline. Its three unchanged
reconstruction windows total **{at['unchanged_reconstruction_span_ms']:.3f} ms
({100*at['unchanged_span_fraction']:.3f}%)**. These windows exclude the observer's
comparison/record/copy work. Even removing all work in those windows gives only
that optimistic avoided-span estimate before cache overhead. It is an
instrumented trace, not a clean performance sample or an implemented speedup.

The trace has one graph launch and an external final synchronization. Captured
nodes are kernels, conditional work, memset and DtoD copies; no host transfers,
callbacks or explicit allocation/free occur inside the profiled work. The
[per-call attribution](../results/validation/vtm/exact-input-observation/unchanged-input-fixed16k-native-v1-attribution.json)
and lossless native trace retain every window and marker match.

## Output and observer validation

The observer's captured test checks unchanged input, a signed-zero bit change,
invalid input, recovery and fresh reset. All seven observation runs succeed.
Six cases have bitwise-identical coordinates, canonical full/material cells,
oriented boundary, roles and work compared with the independently validated
[current controls](larger-active-prefixes.md). The thin-hand control and observer
differ only at point 743's z coordinate, by about **8.0e-19**, while cells,
boundary, roles and work match. Both hand outputs independently pass full-hull,
orientation/empty-sphere, material quality/topology, vertex-link and boundary
self-intersection checks; both VTU exports round-trip. No bitwise-equivalent
hand claim is made.

Two overly strict analysis attempts are retained: v1 required the unused count
in an invalid checkpoint and all numerical change counts to repeat exactly;
v2 additionally required bitwise equality of the hand output. The v3 analysis
checks the actual equality decisions, preserves all numerical differences and
independently validates both hand meshes. No failed runtime invocation is
counted as a successful one.

The [manifest](../results/validation/vtm/exact-input-observation/manifest.json)
pins sources, commands, raw outputs, checker failures, corrected validation,
native trace and shared validators. Reproduce by restoring each artifact's
original `.work/vtm` path and the shared references, with the pinned runtime;
adapt the recorded absolute workspace root as needed. The study driver refuses
overwrites. Use `PYTHONDONTWRITEBYTECODE=1`, the usual regenerable Warp cache,
`PYTHONPATH=.:.work/vtm`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=4`.
Observer/native times are diagnostic. Standalone APIs, defaults and README's
earned Paragram result are unchanged; the wide comparable-quality CPU goal
remains unmet.
'''
p=root/'docs/exact-input-observation.md';assert not p.exists();p.write_text(doc)
print(dict(artifacts=len(meta['files']),source_files=len(meta['runtime_source_sha256']),manifest_sha256=sha(out/'manifest.json')))
