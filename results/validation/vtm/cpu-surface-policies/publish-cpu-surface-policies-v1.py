"""Publish CPU policy qualification and every initial successful/rejected scan control."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/cpu-surface-policies'
assert not out.exists()
subprocess.run([str(root/'.venv/bin/python'),str(w/'verify-cpu-surface-policies-v1.py')],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((w/'cpu-surface-policies-production-v1/result.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==proof['runtime']
prefixes=('cpu-feature-policy-','cpu-distance-policy-','cpu-scan-','cpu-surface-policies-','cpu_mesher_feature_policy_','build-cpu-feature-','check-cpu-feature-','check-cpu-distance-','check-cpu-scan-','check-cpu-surface-policies-','review-cpu-scan-','run-cpu-scan-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(t in p.name for t in ['-issue','-progress','-pr-','-premerge','-merge-','-publication-'])]
files += list(w.glob('cpu-mesh3-feature-policy-v*.cpp'))
files += [w/'cpu-mesh3-target-build.json',Path(__file__).resolve(),w/'verify-cpu-surface-policies-v1.py',w/'write-cpu-surface-policies-doc-v1.py']
for folder in ['cpu-feature-policy-contracts-v1','cpu-feature-policy-contracts-v2','cpu-feature-policy-contracts-v3','cpu-distance-policy-contracts-v1','cpu-scan-feature-pilot-v1','cpu-scan-feature-pilot-v2','cpu-scan-controls-v1','cpu-surface-policies-production-v1']:
    files += [p for p in (w/folder).rglob('*') if p.is_file()]
meta=dict(scope=__doc__,runtime=proof['runtime'],baseline=proof['baseline'],production_source_sha256=proof['production_source_sha256'],files=[],shared_references=[],limits=['No quality-matched CPU speedup. Four of seven single sequential controls validate; different size/fit/quality tradeoffs remain.','Original first failed pilot lost in-memory CPU quota observations; its native stdout/meshes remain, but no clean-quota claim is possible.','Two pilot parallel outputs validate but are throttled; the unthrottled output fails material validation. The mixed-batch review assertion error and correction are retained.','Native binaries are not bundled; build sources/commands and exact binary hashes are recorded. GPU runtime remains pure Warp and unchanged.','CPU feature protection may use weights; no independent unweighted complete-hull claim for material-only exports.','Boundary fitting is sampled quadrature, not a Hausdorff certificate.'],reproduction=['Use the pinned runtime, CGAL 6.2.1 revision44ab808c25399fc6dbd89a6e114f3fd6b5ae8142, GNU11.4, Eigen3/GMP/MPFR and TBB for parallel builds. See benchmarks/cpu/README.md.','Restore files to original paths (adjust absolute workspace paths when relocated), compile fresh named binaries from recorded source/commands, and retain historical reference build separately. Drivers refuse overwrites.','CPU validation: CUDA_VISIBLE_DEVICES empty, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.; no GPU/compiler/heavy-validator overlap during clean native timing.','The exact normalized float32 scan and separate GPU reference are linked below; preserve Stanford/libigl provenance and asset terms.'])
prior=root/'results/validation/vtm/ordered-surface-queries'
refs=[prior/n for n in ['manifest.json','surface-query-order-final-v1-independent.json','surface-query-order-final-v1/armadillo-uniform8192-recovery-after/input.off','surface-query-order-final-v1/armadillo-uniform8192-recovery-after/mesh.npz','surface-query-order-final-v1/armadillo-uniform8192-recovery-after/result.json']]
refs += [root/'results/validation/vtm'/n for n in ['exterior-proof-hierarchy/scanned-armadillo-input/provenance.json','exterior-proof-hierarchy/scanned-armadillo-input/armadillo.off','cpu-mesh3-build.json']]
for p in refs:meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
for p in sorted(set(files)):
    q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
    row=dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
    os.link(p,q);q.chmod(0o444);assert sha(q)==row['sha256'];meta['files'].append(row)
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(dict(files=len(meta['files']),bytes=sum(r['bytes'] for r in meta['files']),manifest_sha256=sha(p)))
