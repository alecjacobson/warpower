"""Bind the exercised native history trace to independently certified timed geometry."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');out=w/'selective-step-history-native-torus-v1-comparison.json';assert not out.exists()
proof=w/'selective-step-history-final-v1-independent.json';d=json.loads(proof.read_text());assert d['complete']
name='torus16384-fixed32';cert=d['cases'][name]['independent']['after'];assert cert['full_hull']['valid']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
helper=w/'check-tangential-production-memory-v3.py';fn=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='oriented_rows');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'))
helper2=w/'check-surface-query-order-final-v1.py';fn=next(n for n in ast.parse(helper2.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='exact_arrays');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper2),'exec'))
a=w/'selective-step-history-final-v1'/f'{name}-after.npz';b=w/'selective-step-history-torus-native-v1.npz';assert sha(a)==cert['mesh_sha256']
comparison=exact_arrays(a,b);assert comparison is not None
native=json.loads((w/'selective-step-history-torus-native-v1.json').read_text());source=json.loads((w/'selective-step-history-production-source-v1.json').read_text());assert native['runtime']==source['runtime']
for p,h in native['source_sha256'].items():
 assert h==(source['source_sha256'][p] if p.startswith('warpower/') else sha(Path(p)))
result=json.loads(a.with_suffix('.json').read_text())
for field in ['counts','rounds','steps','moves','topology']:
 key={'steps':'optimizer_steps','moves':'quality_moves'}.get(field,field);assert native['work'][field]==result[key]
assert native['history_statistics'][0]>0 and native['history_statistics'][1]>0 and native['history_statistics'][3]>0
report=dict(scope=__doc__,runtime=source['runtime'],reference_proof_sha256=sha(proof),reference_mesh_sha256=sha(a),measured_mesh_sha256=sha(b),comparison=comparison,history_statistics=native['history_statistics'],scheduler_work_equal=True,complete=True)
out.write_text(json.dumps(report,indent=2)+'\n');print(report)
