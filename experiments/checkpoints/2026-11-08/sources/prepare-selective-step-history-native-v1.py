"""Add an exercised warm-history native trace, and correct unexecuted memory count."""
import ast,hashlib,json
from pathlib import Path
w=Path(__file__).parent;changes=[]
def save(p,s):
 old=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 ast.parse(s);p.write_text(s);changes.append(dict(path=str(p),previous_sha256=old,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
p=w/'run-selective-step-history-memory-v1.py';s=p.read_text();assert '],39,600)' in s;save(p,s.replace('],39,600)','],37,600)'))
p=w/'run-selective-step-history-qualification-v1.py';s=p.read_text();s=s.replace('[("native-full",', '[("native-history", "run-selective-step-history-torus-native-v1.py"), ("native-full",');save(p,s)
p=w/'check-selective-step-history-qualified-v1.py';s=p.read_text();s=s.replace("jobs=[('native-full',", "jobs=[('native-history',[str(w/'check-selective-step-history-native-torus-v1.py')]),('native-full',");save(p,s)
# The ordinary loop runner's SQL retains all kernels without stage-specific assumptions.
p=w/'run-selective-step-history-torus-native-v1.py';assert not p.exists()
s=(w/'run-selective-step-history-loop-native-v1.py').read_text().replace('selective-step-history-loop-native-v1','selective-step-history-torus-native-v1')
s=s.replace("'-m','pytest','-q','-s',str(w/'test_tangential_production_loop_v2.py')", "str(w/'profile-selective-step-history-torus-v1.py'),'--output',str(stem.with_suffix('.json')),'--dot',str(dot)")
s=s.replace("env=dict(os.environ,WARPOWER_TOPOLOGY_NSYS='1',WARPOWER_TOPOLOGY_DOT=str(dot),WARPOWER_TOPOLOGY_LOOP_OUTPUT=str(w/'selective-step-history-torus-native-v1-frames'))",'env=dict(os.environ)')
s=s.replace('Production uniform8K tangential exhaustion, recovery, invalid target/domain and recovery loop; profiling only','Production16K complete solve with exercised quality-gated warm steps and convergence fallback; profiling only')
save(p,s)
out=w/'selective-step-history-native-preparation-v1.json';assert not out.exists();out.write_text(json.dumps(dict(scope=__doc__,files=changes),indent=2)+'\n')
