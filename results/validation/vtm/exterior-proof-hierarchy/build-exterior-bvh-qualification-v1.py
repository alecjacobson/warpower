from pathlib import Path
w=Path('.work/vtm')
s=(w/'run-larger-prefix-final-v1.py').read_text().split('# Seven original complete-solve controls')[0]
s=s.replace("out=w/'larger-prefix-final-v1';baseline=w/'larger-prefix-baseline'","out=w/'exterior-bvh-final-v1';baseline=w/'exterior-bvh-baseline'")
s=s.replace("'1292 passed' in (w/'larger-prefix-full-suite-v1.log')","'1317 passed' in (w/'exterior-bvh-full-suite-v1.log')")
s=s.replace('df74493c97633324147cb39ef4c41a7d34c7b90a','6f878c7098b525e4cfdbda5e8d2ed45cf8b7e736')
s=s.replace("PYTHONPATH='.'","PYTHONPATH='.'+os.pathsep+str(w)")
s=s.replace("p=subprocess.Popen(cmd,cwd=cwd,env=env,", "runenv=dict(env)\n        if 'run-tangential-topology-gallery-v2.py' in cmd[1]:runenv['WARPOWER_TOPOLOGY_DIAGNOSTICS']=str(out/(name+'-'+phase+'-recovery.json'))\n        p=subprocess.Popen(cmd,cwd=cwd,env=runenv,")
s+='''controls=json.loads((w/'larger-prefix-final-v1/commands.json').read_text())
lookup={r['name']:r for r in controls if r['phase']=='after'}
for name in ['torus16384-fixed32','torus1000']:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=lookup[name]['command'][:]
  cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'))
  cmd[cmd.index('--save-mesh')+1]=str(out/(name+'-'+phase+'.npz'))
  run(name,phase,cmd,cwd)
cases=[('thin-hand-seed7','thin-hand',1000,['--seed','7'],False),
       ('cavity','cavity',1000,[],False),('disconnected','disconnected',1000,[],False),
       ('plate','counterbored-plate',1000,[],False),
       ('armadillo-graded4096','graded-surface',4096,[],False),
       ('armadillo-graded8192-recovery','graded-surface',8192,[],True),
       ('armadillo-uniform8192-recovery','surface',8192,[],True)]
for name,case,n,extra,experimental in cases:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=[str(python)]+([str(w/'run-tangential-topology-gallery-v2.py')] if experimental else ['-m','examples.vtm_gallery'])
  cmd+=['--case',case,'--n',str(n),'--initial','256','--repeats','3',*extra]
  if 'armadillo' in name:
   cmd+=['--input',str(w/'scanned-armadillo-input/armadillo.off'),'--subdivisions','1','--crease-subdivisions','1','--feature-angle','120','--corner-angle','45']
  cmd+=['--output',str(out/(name+'-'+phase))]
  run(name,phase,cmd,cwd)
for name in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
 for phase,cwd in [('before',baseline),('after',root)]:
  cmd=lookup[name]['command'][:];cmd[cmd.index('--output')+1]=str(out/(name+'-'+phase+'.json'));run(name,phase,cmd,cwd)
assert len(rows)==28
assert all(r['valid'] in [None,1] for r in rows)
'''
p=w/'run-exterior-bvh-final-v1.py';assert not p.exists();p.write_text(s)
s=(w/'profile-tangential-topology-uniform-v2.py').read_text().replace('One warmed uniform8K full solve with diagnostic stage markers and unchanged decisions.','Native production hierarchical exterior proof inside private uniform8K recovery; profiling only.')
s=s.replace("Path('.work/vtm/tangential_topology_recovery_v2.py')", "Path('.work/vtm/tangential_topology_recovery_v2.py'),Path('warpower/_exterior_proof_bvh.py'),Path('warpower/_proven_exterior.py')")
p=w/'profile-exterior-bvh-uniform-v1.py';assert not p.exists();p.write_text(s)
s=(w/'run-tangential-topology-uniform-native-v2.py').read_text().replace('tangential-topology-uniform-native-v2','exterior-bvh-uniform-native-v1').replace('profile-tangential-topology-uniform-v2.py','profile-exterior-bvh-uniform-v1.py').replace('Uniform Armadillo 8192 complete replay with private tangential topology recovery; profiling only','Production hierarchical exterior proof with private uniform8K recovery; profiling only')
p=w/'run-exterior-bvh-uniform-native-v1.py';assert not p.exists();p.write_text(s)
