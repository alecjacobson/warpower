"""Strengthen unexecuted qualification guards before collecting any production evidence."""
from pathlib import Path
w=Path('.work/vtm');p=w/'check-larger-prefix-native-loop-v1.py';s=p.read_text();needle="report['complete']=True;";assert s.count(needle)==1
s=s.replace(needle,"assert all(report['recovery_comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])\nassert report['recovery_comparison']['maximum_coordinate_difference']==0\n"+needle);p.write_text(s)
p=w/'run-larger-prefix-qualification-v1.py';s=p.read_text();old="assert len(json.loads((w/'larger-prefix-final-v1/commands.json').read_text()))==34";assert old in s
s=s.replace(old,"commands=json.loads((w/'larger-prefix-final-v1/commands.json').read_text())\nassert len(commands)==34 and all(r['quota_clean'] and r['exit_code'] in [0,1] for r in commands)");p.write_text(s)
