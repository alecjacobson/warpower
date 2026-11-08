"""Verify immutable exact evidence bytes, source bindings and lossless archives."""
import gzip,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/bounded-hull-box';manifest=out/'manifest.json';m=json.loads(manifest.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
supplement=out/'verification-supplement-v2.json'
s=json.loads(supplement.read_text());assert sha(manifest)==s['base_manifest_sha256']
for r in m['files']+s['files']:
    p,q=root/r['path'],root/r['original'];assert sha(p)==sha(q)==r['sha256'],p
    assert p.stat().st_size==r['bytes'] and p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0,p
for group in ['shared_helpers','shared_inputs']:
    for r in m[group]:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256'],r
for p,digest in m['source_sha256'].items():assert sha(root/p)==digest,p
expected={r['path'] for r in m['files']+s['files']}|{str(manifest.relative_to(root)),str(supplement.relative_to(root))}
assert expected=={str(p.relative_to(root)) for p in out.rglob('*') if p.is_file()}
assert subprocess.check_output(['git','diff',m['production_runtime'],'--','warpower','tests','examples','benchmarks','README.md'])==b''
for p in out.rglob('*.gz'):assert gzip.decompress(p.read_bytes())
archives=list(out.glob('*/lossless-archives.json'))+[out/'hull-box-filter-observer-v1-archives.json']
for p in archives:
    for r in json.loads(p.read_text()):
        data=Path(r['path']).read_bytes()
        if 'sha256' in r:assert hashlib.sha256(data).hexdigest()==r['sha256']
        assert hashlib.sha256(gzip.decompress(data)).hexdigest()==r['raw_sha256']
for stem in ['bounded-hull-box-full-native-v1','bounded-hull-box-loop-native-v1','current-torus-profile-v1']:
    k=json.loads((out/(stem+'-kernels.json')).read_text());a=json.loads((out/(stem+'-audit.json')).read_text())
    trace=out/(stem+'.nsys-rep.gz');assert sha(trace)==k['trace_gz_sha256']
    assert hashlib.sha256(gzip.decompress(trace.read_bytes())).hexdigest()==k['trace_raw_sha256']
    assert hashlib.sha256(gzip.decompress((out/(stem+'.dot.gz')).read_bytes())).hexdigest()==a['dot_sha256']
r=dict(files=len(m['files'])+len(s['files']),supplement_sha256=sha(supplement),source_hashes=len(m['source_sha256']),shared_helpers=len(m['shared_helpers']),shared_inputs=len(m['shared_inputs']),manifest_sha256=sha(manifest),all_hashes_verified=True,all_gzip_integrity_and_recorded_raw_export_and_native_hashes_verified=True,original_hardlinks_and_readonly_modes_preserved=True,runtime_unchanged_since=m['production_runtime'])
Path('.work/vtm/verified-bounded-hull-box-publication-v2.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
