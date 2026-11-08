"""Independent complete-mesh validation for the unsuccessful first speed pilot."""
from pathlib import Path
import json
w=Path('.work/vtm');source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0];exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
out=w/'local-flip-pilot-v1';dest=w/'local-flip-pilot-v1-independent.json';assert not dest.exists();report=dict(scope=__doc__,cases={})
for name in ['torus4096','hand7']:
 rows={};raw={}
 for phase in ['before','after']:
  folder=out/(name+'-'+phase);d=json.loads((folder/'result.json').read_text());assert d['valid']==1 and d['error']==0;quota(d);raw[phase]=d
  rows[phase]=dict(median_ms=d['median_additional_replay_ms'],counts=d['counts'],independent=validate(folder/'mesh.npz',folder/'mesh.vtu.gz'))
 with np.load(out/(name+'-before')/'mesh.npz') as x,np.load(out/(name+'-after')/'mesh.npz') as y:rows['comparison']=compare(x,y)
 rows['work_equal']={k:raw['before'][k]==raw['after'][k] for k in ['counts','rounds','steps','moves','quality_rejections','refinement_rejections']};rows['relative_timing_change']=rows['after']['median_ms']/rows['before']['median_ms']-1
 report['cases'][name]=rows;dest.write_text(json.dumps(report,indent=2)+'\n');print(name,rows['comparison'],rows['work_equal'],rows['relative_timing_change'],flush=True)
report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
