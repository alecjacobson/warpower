from pathlib import Path
import ast
w=Path('.work/vtm')
def save(name,s):
 p=w/name;assert not p.exists();ast.parse(s);p.write_text(s)
s=(w/'profile-exterior-bvh-uniform-v1.py').read_text()
s=s.replace('Native production hierarchical exterior proof inside private uniform8K recovery; profiling only.','Native production uniform8K tangential recovery and complete four-round quality solve; profiling only.')
s=s.replace('import tangential_topology_recovery_v2 as experiment\nexperiment.install()','from warpower import _topology_recovery as experiment')
s=s.replace('from warpower import SurfaceDomain','from warpower import SurfaceDomain, VariationalMeshingWorkspace').replace('class MarkedMesher(experiment.Mesher):','class MarkedMesher(VariationalMeshingWorkspace):')
s=s.replace("Path('.work/vtm/tangential_topology_recovery_v2.py')", "Path('warpower/_topology_recovery.py'),Path('warpower/_mesher.py'),Path('warpower/_exterior_band.py'),Path('warpower/_quality_moves.py'),Path('warpower/_extraction.py')")
save('profile-tangential-production-uniform-v1.py',s)
s=(w/'run-exterior-bvh-uniform-native-v1.py').read_text().replace('exterior-bvh-uniform','tangential-production-uniform').replace('Production hierarchical exterior proof with private uniform8K recovery; profiling only','Production bounded tangential recovery and full uniform8K solve; profiling only')
save('run-tangential-production-uniform-native-v1.py',s)
s=(w/'run-exterior-bvh-loop-native-v1.py').read_text().replace('exterior-bvh-loop-native-v1','tangential-production-loop-native-v1')
a=s.index(",str(w/'profile-exterior-bvh-loop-v1.py')");b=s.index('\nstart=time.time()',a)
s=s[:a]+",'-m','pytest','-q','-s',str(w/'test_tangential_production_loop_v2.py')]\nenv=dict(os.environ,WARPOWER_TOPOLOGY_NSYS='1',WARPOWER_TOPOLOGY_DOT=str(dot),WARPOWER_TOPOLOGY_LOOP_OUTPUT=str(w/'tangential-production-loop-native-v1-frames'))"+s[b:]
s=s.replace('subprocess.Popen(cmd,stdout=log', 'subprocess.Popen(cmd,env=env,stdout=log')
s=s.replace('Hierarchical exterior proof in a production thin-hand seed/failure/recovery loop; profiling only','Production uniform8K tangential exhaustion, recovery, invalid target/domain and recovery loop; profiling only')
save('run-tangential-production-loop-native-v1.py',s)
s=(w/'check-tangential-production-loop-v2.py').read_text().replace("folder=w/'tangential-production-loop-v2'", "folder=w/'tangential-production-loop-native-v1-frames'").replace("dest=w/'tangential-production-loop-v2-independent.json'", "dest=w/'tangential-production-loop-native-v1-independent.json'")
save('check-tangential-production-loop-native-v1.py',s)
