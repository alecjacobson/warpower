"""Focused padded and allocation-lifetime checks of the changed query path."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root=Path.cwd(); w=root/'.work/vtm'; python=root/'.venv/bin/python'
assert '851 passed' in (w/'current-selection-full-suite.log').read_text()
assert not (w/'current-selection-memory-v2-commands.json').exists()
rows=[]
for name,test in [('selection-oracle',['tests/test_current_selection.py']),('controls-rollback',['tests/test_vtm_controls.py::test_device_producer_changes_budget_and_mode_with_invalid_recovery','tests/test_selective.py::test_rejected_quality_updates_restore_original_geometry_and_statistics','tests/test_containment_selection.py'])]:
    for mode in ['padded','lifetime']:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
        assert env.get('WARP_CACHE_PATH')=='/opt/warpower-generated-cache'
        options=[]
        if mode=='lifetime':
            env['WARPOWER_AUDIT_DISABLE_POOL']='1'
            options=['--check-api-memory-access','yes','--track-stream-ordered-races','use-after-free']
        cmd=['/usr/local/cuda-12.9/bin/compute-sanitizer','--tool','memcheck','--padding','32','--error-exitcode','99',*options,str(python),str(w/'large-reuse-isolated-test.py'),*test]
        log=w/f'current-selection-v2-{name}-{mode}.log';assert not log.exists();start=time.time()
        with log.open('w') as stream:
            p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=124
        text=log.read_text()
        row=dict(name=name,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,disable_pool=mode=='lifetime',zero_memcheck_errors='ERROR SUMMARY: 0 errors' in text)
        rows.append(row);(w/'current-selection-memory-v2-commands.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
        assert code==0 and row['zero_memcheck_errors'],row
