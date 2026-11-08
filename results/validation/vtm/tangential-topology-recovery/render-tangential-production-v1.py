"""CPU-only cutaways of the independently validated production timed meshes."""
import hashlib,json
from pathlib import Path
from examples.vtm_render import render
w=Path('.work/vtm');d=json.loads((w/'tangential-production-final-v1-independent.json').read_text());assert d['complete']
record=[]
for kind in ['uniform','graded']:
 name=f'armadillo-{kind}8192-recovery';p=w/'tangential-production-final-v1'/(name+'-after')/'mesh.npz';out=w/f'tangential-production-{kind}8192-cutaway-v1.png';assert not out.exists()
 render(p,out,axis=2,fraction=0.6,up_axis='y',elevation=15,azimuth=-20,title=f'{kind.capitalize()} Armadillo · 8K tetrahedral mesh',credit='Stanford Computer Graphics Laboratory · libigl tutorial-data distribution')
 record.append(dict(case=name,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),output=str(out),output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256(Path('examples/vtm_render.py').read_bytes()).hexdigest(),camera=dict(axis=2,fraction=0.6,up_axis='y',elevation=15,azimuth=-20)))
p=w/'tangential-production-cutaways-v1.json';assert not p.exists();p.write_text(json.dumps(dict(scope=__doc__,renders=record),indent=2)+'\n');print(record)
