from pathlib import Path
w=Path('.work/vtm')
# These v2 drivers have not been executed yet.
names="['torus16384-fixed32','graded-torus16384','torus8192-progressive','torus8192-fixed32']"
p=w/'run-prepared-new-cell-pilot-v2.py';s=p.read_text().replace("['torus16384-fixed32','graded-torus16384']",names).replace('assert len(rows)==4','assert len(rows)==8');p.write_text(s)
p=w/'check-prepared-new-cell-pilot-v2.py';s=p.read_text().replace("['torus16384-fixed32','graded-torus16384']",names).replace('assert len(commands)==4','assert len(commands)==8');p.write_text(s)
for prefix in ['bench','run','check']:
 p=w/(prefix+'-prepared-new-cell-grid-v2.py');assert not p.exists()
 s=(w/(prefix+'-hierarchical-split-grid-v2.py')).read_text().replace('hierarchical-split-grid-v2','prepared-new-cell-grid-v2').replace('hierarchical-split-pilot-v2','prepared-new-cell-pilot-v2').replace('hierarchical_split_v2','prepared_new_cell_integration_v2').replace('== 10','== 8')
 p.write_text(s)
