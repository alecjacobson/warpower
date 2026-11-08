"""Optional private full-route reuse beyond 4096; ordinary public mesher harness."""
import hashlib
import json
from pathlib import Path
import sys

candidate = '--candidate' in sys.argv
if candidate:
    sys.argv.remove('--candidate')
    import extended_reuse_v1
    extended_reuse_v1.install()
from benchmarks.mesher import main
main()
path = Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.route.json')
assert not path.exists()
files = [Path(__file__)]
if candidate:
    files.append(Path(extended_reuse_v1.__file__))
metadata = dict(candidate=candidate, prototype_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
if candidate:
    metadata.update(original_class_module_sha256=extended_reuse_v1.source_sha256, substituted_module_sha256=extended_reuse_v1.candidate_sha256, restored_ast_matches_original=True)
path.write_text(json.dumps(metadata,indent=2)+'\n')
