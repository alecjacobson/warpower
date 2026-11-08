"""Sequential device memory/capture gates, after the clean timing batch is terminal."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';python=root/'.venv/bin/python';dest=w/'large-stackless-qualification-v1.json';assert not dest.exists()
assert len(json.loads((w/'large-stackless-final-v1/commands.json').read_text()))==22
jobs=[('full',[str(python),str(w/'run-large-stackless-native-v1.py')],650),('loop',[str(python),str(w/'run-large-stackless-native-v1.py'),'--loop'],650),('memory',[str(python),str(w/'run-large-stackless-memory-v1.py')],19000)]
rows=[]
for name,cmd,timeout in jobs:
    path=w/f'large-stackless-qualification-v1-{name}.log';assert not path.exists();start=time.time()
    with path.open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=timeout)
    rows.append(row);dest.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True);assert code==0,row
