"""Independently validate all retained seed1 repeat meshes and identify material alternatives."""
import json
from pathlib import Path
w=Path('.work/vtm')
source=(w/'check-surface-bound-reuse-final-v2.py').read_text().split("out=w/'surface-bound-reuse-final'")[0]
exec(compile(source,str(w/'check-surface-bound-reuse-final-v2.py'),'exec'))
from examples.vtm_inputs import write_off
from benchmarks.feature_quality import measure as feature_measure

dest=w/'surface-query-cache-repeats-independent-v1.json';assert not dest.exists()
final=json.loads((w/'surface-query-cache-final-independent-v2.json').read_text())
refs={phase:np.load(w/f'surface-query-cache-final/hand-seed1-{phase}/mesh.npz') for phase in ['before','after']}
result=dict(scope=__doc__,phases={},complete=False)
for phase,expected_runtime in [('before','07aa4c78fc69fdeae468db457f455c953ca84c06'),('after','d8c0cab8213eaf9b3bb0638fa97dbf7ebbf8cda3')]:
    folder=w/f'surface-query-cache-repeat-{phase}-v1'
    raw=json.loads((folder/'result.json').read_text());assert raw['runtime']==expected_runtime and len(raw['replays'])==8
    final_raw=json.loads((w/f'surface-query-cache-final/hand-seed1-{phase}/result.json').read_text())
    records=[];counts={'before':0,'after':0};result['phases'][phase]=dict(runtime=expected_runtime,result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),replays=records,material_alternative_counts=counts)
    for row in raw['replays']:
        path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        assert row['valid']==[1] and row['error']==[0]
        work={k:row[k]==final_raw[k] for k in ['counts','rounds','steps','moves','topology']}
        work.update({k:row[k]==[final_raw[k]] for k in ['quality_rejections','refinement_rejections']});assert all(work.values())
        independent=validate(path)
        with np.load(path) as data:
            comparisons={key:compare(ref,data) for key,ref in refs.items()}
            labels=[key for key,c in comparisons.items() if c['material_tets_equal'] and c['oriented_boundary_equal']]
            assert len(labels)==1,comparisons
            label=labels[0];counts[label]+=1
            assert comparisons[label]['roles_equal'] and comparisons[label]['maximum_coordinate_difference']<1e-8
            for key in ['input_points','input_faces']:np.testing.assert_array_equal(data[key],refs[label][key])
            with tempfile.TemporaryDirectory(prefix='query-repeat-boundary-',dir=scratch) as tmp:
                off=Path(tmp)/'boundary.off';write_off(off,data['points'],data['boundary'])
                text=subprocess.run([str(w/'surface-reference'),str(off)],text=True,capture_output=True,check=True).stdout
                assert int(text.split()[2])==0,text
            independent['boundary_self_intersection_pairs']=0
            features=feature_measure(data['points'],data['boundary'],data['roles'],np.unique(data['tetrahedra']),data['input_points'],data['input_faces'],feature_angle=45,corner_angle=45,samples_per_edge=16,distance_query=NativeSurfaceDistance(w/'surface-distance',scratch))
            assert all(c['exact_coordinate_match'] for c in features['corners'])
        record=dict(index=row['index'],sha256=row['sha256'],material_alternative=label,comparisons=comparisons,recorded_work_equal=work,independent=independent,features=features,features_equal_to_final_reference=features==final['cases']['hand-seed1'][label]['features'])
        records.append(record);dest.write_text(json.dumps(result,indent=2)+'\n')
        print(phase,row['index'],label,record['features_equal_to_final_reference'],flush=True)
    assert all(n>0 for n in counts.values()),counts
result['complete']=True
result['interpretation']='Both material triangulations occur in both unchanged baseline and candidate repeated fresh solves. All sixteen own-frame outputs independently validate. This establishes pre-existing output variability, not equal distributions or a unique causal source of floating-point differences.'
dest.write_text(json.dumps(result,indent=2)+'\n')
print({phase:row['material_alternative_counts'] for phase,row in result['phases'].items()},flush=True)
