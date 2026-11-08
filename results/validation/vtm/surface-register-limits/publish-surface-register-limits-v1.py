"""Publish the rejected register-limit probe without changing production kernels."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/surface-register-limits'
assert not out.exists()
source=json.loads((w/'surface-register-limits-source-v1.json').read_text())
r=json.loads((w/'surface-register-limits-summary-v1.json').read_text());assert r['complete'] and not r['adopted'] and not r['complete_solve_measured']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['baseline']
assert not subprocess.check_output(['git','diff','HEAD','--','warpower','tests','benchmarks','examples','README.md','pyproject.toml'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'warpower/_ordered_surface_queries.py')==source['original_source_sha256']
table='| Static cold batch | Mode | Original, ms | Limit 96, ms | Limit 128, ms | Limit 192, ms |\n|---|---|---:|---:|---:|---:|\n'
for name,c in r['cases'].items():
    for row in c['rows']:
        if row['cache']!='cold':continue
        t=row['ms'];table+=f"| {name} | {row['mode']} | {t['before']:.3f} | {t['r96']:.3f} | {t['r128']:.3f} | {t['r192']:.3f} |\n"
doc='''# Surface-classifier register limits: no adoption

Warp's built-in register limits do not provide a useful broad improvement for
the currently expensive ordered full/selection classifiers. Limits of 96 and 128
regress several large cold batches by about 10–27%; 192 is essentially flat there.
Production kernels, public APIs and the README remain unchanged. This closes the
bounded probe in [#250](https://github.com/alecjacobson/warpower/issues/250), not
the wider query-optimization or comparable-quality CPU speedup goals.

The [ordered-query baseline](ordered-surface-queries.md) native uniform 8K trace
attributes 681.7 ms to ordered selection and 400.4 ms to ordered full classification,
both at 255 registers/thread with block size 32. These diagnostic times motivated
the experiment; register count alone does not diagnose an occupancy bottleneck.

## Experiment and measurements

Baseline main `0a2ce5e4ed48`, NVIDIA L40, Warp 1.19.0+cu12. Three private modules copy
the existing classifier arithmetic exactly by AST, changing only the
`cuda_max_registers` decorator option. A private installer selects kernel objects
before capture. Production ordering, cache, key generation and host launch code
are unchanged. The installed Warp API/native-library bytes match their package
RECORD; no Warp source or custom CUDA implementation is added.

The compiled kernels use the requested 96/128/192 registers/thread. Their reported
local memory is respectively 36,704–36,784, 36,496–36,552 and 36,240–36,296 bytes
per thread, versus 35,952–36,008 for the original three kernels. These properties
describe compiled resources; they do not prove a measured occupancy or bandwidth
change.

Each case is an immutable independently validated final mesh. Twenty warmed GPU
samples per row include key generation, sorting, queries and cache rebuilding;
cold rows also clear the cache. Warm rows repeatedly reuse unchanged input.
All 96 rows have clean CPU-quota windows, and every candidate's successful output
is bitwise equal to its original reference. Setup and host verification are
outside timing. These are static batches, not complete solves.

'''+table+'''
The 1K torus stays below the ordering gate and uses the unchanged scalar path;
it does not exercise capped kernels during its timing rows. Warm-cache scan rows
can improve by roughly 6% with 96/128, but their small absolute costs and the cold
regressions do not establish a useful complete-solve change. Retained classification
has some static improvements, including graded 8K at 96/128. The motivating uniform 8K
native trace contains no executed retained-classifier kernel, so those observations
cannot be substituted for a gain in that full solve. This probe does not rule out
a separately justified targeted policy on other workloads.

## Validation and limits

All 19 existing ordered-query cases pass for each cap: 57 case executions, including
forced ordering at extreme scales, changing live counts/errors, nondefault streams
and inherited cache contracts. The first static harness mistakenly kept the old
two-variant loop labels and stopped with `KeyError` after one baseline row. It had
not timed or compared a candidate. That source, log and partial result remain;
v2 corrects only the phase iteration and reuses the unchanged hash-bound contracts.

No candidate was adopted or measured through a complete solve. There is no new
candidate native capture, sanitizer/lifetime or full-mesh geometry audit claimed.
The current production qualification remains unchanged. The
[manifest](../results/validation/vtm/surface-register-limits/manifest.json)
retains all variants, compiled properties, raw samples, source hashes, the failed
harness and the correction. [#197](https://github.com/alecjacobson/warpower/issues/197)
and [#40](https://github.com/alecjacobson/warpower/issues/40) remain open.
'''
p=root/'docs/surface-register-limits.md';assert not p.exists();p.write_text(doc)
p=root/'docs/ordered-surface-queries.md';s=p.read_text();s+='\nA subsequent [register-limit probe](surface-register-limits.md) retained the\noriginal kernels: lower caps regressed important cold batches and no complete-solve\ngain was established.\n';p.write_text(s)
prefixes=('surface-register-limits-','surface_register_limit_','surface_register_limits_','build-surface-register-limits-','test-surface-register-limits-','profile-surface-register-limits-','run-surface-register-limits-','summarize-surface-register-limits-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-issue','-pr-','-merge-','-premerge','-publication-'])]
for folder in ['surface-register-limits-v1','surface-register-limits-v2']:files += [p for p in (w/folder).rglob('*') if p.is_file()]
files += [Path(__file__).resolve(),w/'verify-surface-register-limits-v1.py']
meta=dict(scope=__doc__,baseline=source['baseline'],files=[],shared_references=[],production_source_unchanged=True,adopted=False,limits=r['limits'],reproduction=['Restore original artifact paths and referenced current meshes; use pinned baseline, NVIDIA L40 and Warp 1.19.0+cu12 with no modified package source.','Use .venv/bin/python, PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process; no heavy CPU work during clean timing.','Build variants, run 19 contracts percap, then corrected v2 static runner. Retain failed v1 harness as evidence; do not overwrite outputs.','No full-solve/adoption gates are bypassed by the static measurements.'])
prior=root/'results/validation/vtm/ordered-surface-queries'
refs=[prior/n for n in ['manifest.json','surface-query-order-final-v1-independent.json','surface-query-order-uniform-native-v1-kernels.json','surface-query-order-final-v1/torus1000-after.npz','surface-query-order-final-v1/torus16384-fixed32-after.npz','surface-query-order-final-v1/armadillo-uniform8192-recovery-after/mesh.npz','surface-query-order-final-v1/armadillo-graded8192-recovery-after/mesh.npz']]
for p in refs:meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
for p in sorted(set(files)):
    q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
    row=dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size);os.link(p,q);q.chmod(0o444);assert sha(q)==row['sha256'];meta['files'].append(row)
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(dict(files=len(meta['files']),manifest_sha256=sha(p)))
