from pathlib import Path
w=Path('.work/vtm');s=(w/'run-exterior-bvh-final-v2.py').read_text()
s=s.replace('exterior-bvh-final-v2','tangential-production-final-v1').replace('exterior-bvh-baseline','tangential-production-baseline').replace('exterior-bvh-full-suite-v1','tangential-production-full-suite-v1').replace('1317 passed','1326 passed').replace('6f878c7098b525e4cfdbda5e8d2ed45cf8b7e736','ceb5ea1420b4c9e9a7b16c89a0432e567962b376')
a=s.index('# Reuse the completed');b=s.index('def sha(p):');s=s[:a]+s[b:]
s=s.replace("if 'run-tangential-topology-gallery-v2.py' in cmd[1]", "if 'run-tangential-production-gallery-v1.py' in cmd[1]")
s=s.replace("for name in ['torus1000']", "for name in ['torus16384-fixed32','torus1000']")
s=s.replace("cmd=[str(python)]+([str(w/'run-tangential-topology-gallery-v2.py')] if experimental else ['-m','examples.vtm_gallery'])", "cmd=[str(python),str(w/'run-tangential-production-gallery-v1.py')]")
s=s.replace("assert all(r['valid'] in [None,1] for r in rows)", "assert sum(r['valid']==0 for r in rows)==2\nassert all(r['phase']=='before' and '8192' in r['name'] for r in rows if r['valid']==0)")
p=w/'run-tangential-production-final-v1.py';assert not p.exists();p.write_text(s)
