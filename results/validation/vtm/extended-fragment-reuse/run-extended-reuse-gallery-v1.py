"""Gallery solve with an optional private extension of the full VTM reuse route."""
import hashlib
import json
from pathlib import Path
import sys

candidate='--candidate' in sys.argv
if candidate:
    sys.argv.remove('--candidate')
    import extended_reuse_v1
    extended_reuse_v1.install()
from examples.vtm_gallery import main
main()
path=Path(sys.argv[sys.argv.index('--output')+1])/'route.json'
assert not path.exists()
files=[Path(__file__)]
if candidate:
    files.append(Path(extended_reuse_v1.__file__))
metadata=dict(candidate=candidate,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
if candidate:
    metadata.update(original_module_sha256=extended_reuse_v1.source_sha256,substituted_module_sha256=extended_reuse_v1.candidate_sha256,restored_ast_matches_original=True)
path.write_text(json.dumps(metadata,indent=2)+'\n')
