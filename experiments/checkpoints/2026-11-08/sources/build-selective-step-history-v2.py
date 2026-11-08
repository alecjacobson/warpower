"""Gate private history on fresh quality-only failures; retain full default fallback."""
import ast
import hashlib
import json
from pathlib import Path

w=Path(".work/vtm");old=w/"selective_step_history_v1.py";new=w/"selective_step_history_v2.py"
assert not new.exists()
proof=json.loads((w/"selective-step-history-pilot-v1-independent-v2.json").read_text());assert proof["complete"]
assert proof["cases"]["armadillo-uniform8192-recovery"]["timing"]["reduction_fraction"]<0
s=old.read_text()


def replace(a,b,count=1):
    global s
    assert s.count(a)==count,(a,s.count(a))
    s=s.replace(a,b)


replace('Private bounded selective alpha history; not a production implementation.','Private quality-gated selective alpha history; not a production implementation.')
replace('    statistics: wp.array[int], max_backtracks: int,','    statistics: wp.array[int], max_backtracks: int, selection_rejections: wp.array[int],')
replace('    history[3]=0  # this iteration starts below the default scale','    history[3]=0  # this iteration starts below the default scale\n    history[4]=1  # every observed rejection in this search was quality-only\n    history[5]=selection_rejections[0]')
replace('    statistics: wp.array[int], damping: wp.float64,','    statistics: wp.array[int], damping: wp.float64,\n    selection_rejections: wp.array[int], selection_last_error: wp.array[int],')
replace('    warm=mode[0]==1 and history[3]!=0 and history[2]==0','''    fresh_quality_failure=(selection_rejections[0]>history[5] and selection_last_error[0]==4
        and mesh_error[0]==0 and boundary_error[0]==0)
    history[5]=selection_rejections[0]
    if accepted[0]==0 and not fresh_quality_failure:
        history[4]=0
    warm=mode[0]==1 and history[3]!=0 and history[2]==0''')
replace('    retry_exhaustion=warm and accepted[0]==0 and backtracks[0]>=max_backtracks','''    retry_exhaustion=warm and accepted[0]==0 and backtracks[0]>=max_backtracks and fresh_quality_failure
    retry_nonquality=warm and accepted[0]==0 and not fresh_quality_failure''')
replace('        history[2]=1\n        statistics[1]+=1','        history[2]=1\n        history[4]=1\n        statistics[1]+=1')
replace('    if retry_exhaustion:', '    if retry_exhaustion or retry_nonquality:')
replace('        history[2]=1\n        statistics[2]+=1','''        history[2]=1
        history[4]=1
        statistics[2]+=int(retry_exhaustion)
        statistics[4]+=int(retry_nonquality)''')
replace('            history[0]=1\n            history[1]=backtracks[0]','            history[0]=history[4]\n            history[1]=backtracks[0]')
replace('self.private_step_history=wp.zeros(4,dtype=int,device=domain.device)','self.private_step_history=wp.zeros(6,dtype=int,device=domain.device)')
replace('self.private_step_statistics=wp.zeros(4,dtype=int,device=domain.device)','self.private_step_statistics=wp.zeros(5,dtype=int,device=domain.device)')
replace('self.private_step_statistics,self.max_backtracks]','self.private_step_statistics,self.max_backtracks,self.selection.rejections]')
replace('self.private_step_statistics,self.damping]','self.private_step_statistics,self.damping,self.selection.rejections,self.selection.last_error]')
tree=ast.parse(s);original=ast.parse(old.read_text())
for name in ["original_start_attempt","original_finish_attempt"]:
    a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b)
new.write_text(s)
p=w/"selective-step-history-build-v2.json";assert not p.exists()
p.write_text(json.dumps(dict(scope=__doc__,source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),target=str(new),target_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),original_guard_functions_ast_identical=True,reason='V1 uniform scan regressed4.98% with one exhausted warm-search default retry; allbaseline failures were selectionerror16/count0/reason3. V2 only seeds history after fresh quality-only rejection searches and immediately retriesdefault after a different warm failure. No speedup or qualification claimed yet.'),indent=2)+'\n')
