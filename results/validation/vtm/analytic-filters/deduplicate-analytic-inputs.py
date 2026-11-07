"""Remove only byte-identical private copies of a retained persistent input."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');canonical=w/'scanned-armadillo-input/armadillo.off'
assert canonical.is_file() and not canonical.is_symlink()
data=canonical.read_bytes();digest=hashlib.sha256(data).hexdigest()
assert digest=='b4e07d0cea859ec879b46ed632d11206db0d63705c2d4399edd88ead1e1a601a'
rows=[]
for folder in ['analytic-face-broad','analytic-face-broad-recheck','analytic-face-broad-recheck2','analytic-final']:
    for p in sorted((w/folder).glob('*/input.off')):
        if p.read_bytes()!=data:continue
        rows.append(dict(removed_exact_duplicate=str(p),retained_persistent_original=str(canonical),bytes=p.stat().st_size,sha256=digest))
        p.unlink()
(w/'analytic-input-deduplication.json').write_text(json.dumps(rows,indent=2)+'\n')
print('verified_duplicates',len(rows),'bytes',sum(r['bytes'] for r in rows))
