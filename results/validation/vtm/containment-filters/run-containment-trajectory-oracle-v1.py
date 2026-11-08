"""Gallery wrapper for untimed duplicate-predicate decision auditing."""
import hashlib,json,sys
from pathlib import Path
from containment_trajectory_oracle_v1 import install
stats=install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 result=stats.numpy().tolist();paths=[Path(__file__),Path('.work/vtm/containment_trajectory_oracle_v1.py'),Path('.work/vtm/containment_exact_baseline_v1.py'),Path('warpower/_containment.py')]
 report=dict(scope='Instrumented correctness only, not performance. Every eligible cell in every active containment call computes both original integer and filtered whole-cell decisions. Candidate decisions govern unchanged transaction.',columns=['eligible_cells','mismatched_decisions','exact_positive','filtered_positive'],statistics=result,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 (out/'decision-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
 assert result[0]>0 and result[1]==0 and result[2]==result[3],report
