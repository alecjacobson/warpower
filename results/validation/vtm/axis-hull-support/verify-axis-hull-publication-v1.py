"""Verify the closed evidence bundle before committing or merging it."""
import gzip,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/axis-hull-support';manifest=out/'manifest.json';m=json.loads(manifest.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert len(m['files'])>200
for r in m['files']:
 p,q=root/r['path'],root/r['original'];assert sha(p)==sha(q)==r['sha256'],p
 assert p.stat().st_size==r['bytes'] and p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0,p
for group in ['shared_helpers','shared_inputs']:
 for r in m[group]:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256'],r
for p,digest in m['source_sha256'].items():assert sha(root/p)==digest,p
expected={r['path'] for r in m['files']}|{str(manifest.relative_to(root))}
assert expected=={str(p.relative_to(root)) for p in out.rglob('*') if p.is_file()}
assert subprocess.check_output(['git','diff',m['production_runtime'],'--','warpower','tests','examples','benchmarks','README.md'])==b''
for group in ['axis-hull-final']:
 for r in json.loads((out/group/'lossless-archives.json').read_text()):
  raw=gzip.decompress(Path(r['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==r['raw_sha256']
for p in (out/'axis-hull-pilot-v1').glob('*/*.gz'):
 assert gzip.decompress(p.read_bytes())
for r in json.loads((out/'axis-hull-observation-archives-v1.json').read_text()):
 raw=gzip.decompress(Path(r['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==r['raw_sha256']
for stem,label in [(name,name) for name in ['axis-hull-full-native-v1','axis-hull-loop-native-v1']]:
 k=json.loads((out/(label+'-kernels.json')).read_text());a=json.loads((out/(label+'-audit.json')).read_text())
 trace=out/(stem+'.nsys-rep.gz');assert sha(trace)==k['trace_gz_sha256']
 assert hashlib.sha256(gzip.decompress(trace.read_bytes())).hexdigest()==k['trace_raw_sha256']
 assert hashlib.sha256(gzip.decompress((out/(stem+'.dot.gz')).read_bytes())).hexdigest()==a['dot_sha256']
r=dict(files=len(m['files']),source_hashes=len(m['source_sha256']),shared_helpers=len(m['shared_helpers']),shared_inputs=len(m['shared_inputs']),manifest_sha256=sha(manifest),all_hashes_verified=True,final_and_observation_raw_export_hashes_and_native_traces_verified=True,pilot_gzip_integrity_verified=True,original_hardlinks_and_readonly_modes_preserved=True,runtime_unchanged_since=m['production_runtime'])
Path('.work/vtm/verified-axis-hull-publication-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
