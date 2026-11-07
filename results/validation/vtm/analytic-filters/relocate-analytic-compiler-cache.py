"""Move only reproducible compiler cache artifacts, never source/history/results."""
import hashlib,json,shutil
from pathlib import Path
source=Path('/home/horde/.cache/warp/1.19.0');dest=Path('/opt/warpower-generated-cache/1.19.0')
assert not dest.exists();dest.mkdir(parents=True)
records=[]
for p in sorted(source.iterdir()):
    if not p.is_dir() or p.is_symlink():continue
    target=dest/p.name;shutil.copytree(p,target)
    files=[]
    for f in sorted(p.rglob('*')):
        if not f.is_file():continue
        assert not f.is_symlink()
        rel=f.relative_to(p);a=hashlib.sha256(f.read_bytes()).hexdigest()
        assert a==hashlib.sha256((target/rel).read_bytes()).hexdigest()
        files.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=a))
    remove=p.name.startswith(('wp_warpower.','wp_tests.','wp_facet_','wp_analytic_','wp_test_analytic','wp_test_facet_','wp_make_cells_'))
    records.append(dict(original=str(p),regenerable_cache_copy=str(target),original_cache_removed=remove,files=files))
    if remove:shutil.rmtree(p)
record=dict(scope='Only regenerable Warp compiler artifacts copied to ephemeral cache; no project source/Git/unique results moved or removed. Authoritative source remains persistent. Use WARP_CACHE_PATH=/opt/warpower-generated-cache for subsequent jobs.',initial_attempt='Creating /opt/warpower-generated-cache failed with PermissionError before any copy/deletion. The first graded-pair recheck consequently failed during Warp initialization, with no timing. Logs retained. Created only the cache directory via sudo install, owned by horde, before this successful retry.',records=records)
Path('.work/vtm/analytic-compiler-cache-relocation.json').write_text(json.dumps(record,indent=2)+'\n')
print('verified_cache_files',sum(len(r['files']) for r in records),'removed_regenerable_bytes',sum(f['bytes'] for r in records if r['original_cache_removed'] for f in r['files']))
