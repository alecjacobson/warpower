"""Install private recovery and preserve diagnostics even when the full mesher fails."""
import hashlib,json,os,sys
from pathlib import Path
from tangential_topology_recovery_v2 import install
instances=install()
from exterior_plane_bvh_v1 import install as install_proof
install_proof(block_dim=256)
from examples.vtm_gallery import main
try:main()
finally:
    for i,m in enumerate(instances):
        r=m.topology_recovery
        if r is None:continue
        info=dict(scope='Private bounded topology recovery; complete mesher result is recorded separately.',
                  source_sha256=hashlib.sha256(Path(__file__).with_name('tangential_topology_recovery_v2.py').read_bytes()).hexdigest())
        info['proof_source_sha256']=hashlib.sha256(Path(__file__).with_name('exterior_plane_bvh_v1.py').read_bytes()).hexdigest()
        for name in ['chosen','invoked','accepted','attempts','error','status','rejection_reason','metrics','previous_metrics','topology','previous_topology','limit','frame','proposed_position','references_rejected','last_errors']:
            info[name]=getattr(r,name).numpy().tolist()
        p=Path(os.environ['WARPOWER_TOPOLOGY_DIAGNOSTICS']);assert not p.exists();p.write_text(json.dumps(info,indent=2)+'\n');print('TOPOLOGY_RECOVERY',info,flush=True)
