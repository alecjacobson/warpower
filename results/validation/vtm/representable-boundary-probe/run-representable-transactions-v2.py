"""Sequential private core pilots; retain failures and every resulting mesh."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';records=[];path=w/'representable-transactions-v2-commands.json';assert not path.exists()
for case in ['old-hand','current-hand','torus']:
    for mode in ['baseline','candidate']:
        stem=w/f'representable-transaction-{case}-{mode}-v2'
        assert not stem.with_suffix('.log').exists()
        cmd=[str(root/'.venv/bin/python'),str(w/'representable_boundary_transaction_v2.py'),'--case',case,'--mode',mode,'--steps','10','--output',str(stem.with_suffix('.json'))]
        start=time.time()
        with stem.with_suffix('.log').open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=124
        row=dict(case=case,mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=180)
        records.append(row);path.write_text(json.dumps(records,indent=2)+'\n');print(row,flush=True)
        assert code==0,row
