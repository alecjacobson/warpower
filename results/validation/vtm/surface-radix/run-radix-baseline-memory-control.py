"""Bounded diagnostic control of the original combined mesher sequence."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';cwd=w/'extraction-geometry';python=root/'.venv/bin/python'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()=='18b18f566ca1bd0c9a4cf5e96caf623ab009f07c'
assert not subprocess.check_output(['git','status','--short'],cwd=cwd,text=True)
cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',str(python),'-m','benchmarks.memcheck','--stack-bytes','131072','--','-q','-s','-o','faulthandler_timeout=120','-o','faulthandler_exit_on_timeout=true']
cmd += [f'tests/test_mesher.py::test_complete_pipeline_samples_grows_fits_and_preserves_domain_topology[{case}]' for case in ['cube','torus','cavity']]
start=time.time()
with (w/'radix-baseline-mixed-memory.log').open('w') as f:r=subprocess.run(cmd,cwd=cwd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
report=dict(source='18b18f566ca1bd0c9a4cf5e96caf623ab009f07c',command=cmd,cwd=str(cwd),exit_code=r.returncode,wall_seconds=time.time()-start,scope='Original first three combined tests; per-test 120-second diagnostic timeout with stack dump. Nonzero exit is preserved, never counted as a pass.')
(w/'radix-baseline-mixed-memory-command.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
