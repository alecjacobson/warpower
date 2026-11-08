"""Integrate only the measured exact child-push filter after complete pilot parity."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

w=Path('.work/vtm'); baseline='55f356c7aa8cf91d5143b1f392e3b96038c6462d'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==baseline
assert not subprocess.check_output(['git','status','--porcelain'])
proof_path=w/'surface-push-pruning-pilot-v1-independent.json'
proof=json.loads(proof_path.read_text())
assert proof['complete'] and proof['strict_trajectory_parity'] and proof['exact_geometry_transfers']==8
assert all(r['timing']['additional_pool_bytes']==0 and r['timing']['after_ms']<r['timing']['before_ms'] for r in proof['cases'].values())
for name in ['surface-push-pruning-contracts-v1.json','surface-push-pruning-cache-contracts-v1.json']:
    report=json.loads((w/name).read_text()); assert report['complete'] and report['passed']
    for p,h in report['source_sha256'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
path=Path('warpower/_surface_bvh.py'); original=path.read_text()
old='''            stack[size] = left
            stack[size + 1] = right
            bounds[size] = left_distance
            bounds[size + 1] = right_distance
            size += 2'''
new='''            # Keep the original rejection comparison, including unordered NaNs.
            # The best distance only decreases; a rejected child cannot revive.
            if not (left_distance >= result.distance2):
                stack[size] = left
                bounds[size] = left_distance
                size += 1
            if not (right_distance >= result.distance2):
                stack[size] = right
                bounds[size] = right_distance
                size += 1'''
assert original.count(old)==1
candidate=original.replace(old,new)
assert candidate.replace(new,old)==original
def function(source,name): return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
a=function(candidate,'closest_point'); b=function((w/'surface_push_pruning_v1.py').read_text(),'pruned_closest_point'); b.name='closest_point'
assert ast.dump(a)==ast.dump(b)
path.write_text(candidate)
report=dict(scope=__doc__,baseline=baseline,only_changed_source=str(path),before_sha256=hashlib.sha256(original.encode()).hexdigest(),after_sha256=hashlib.sha256(candidate.encode()).hexdigest(),candidate_ast_matches_measured_private_function=True,all_other_source_bytes_preserved_by_reversal=True,pilot_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out=w/'surface-push-pruning-integration-v1.json'; assert not out.exists(); out.write_text(json.dumps(report,indent=2)+'\n'); print(report)
