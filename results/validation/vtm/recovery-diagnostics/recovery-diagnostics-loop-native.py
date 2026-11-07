"""One device loop: changing material selection and full-reference consumers."""
import argparse,json
from pathlib import Path
import warp as wp
import warpower
from tests.test_current_selection import recovered_selection

@wp.kernel
def produce(base:wp.array[wp.vec3],points:wp.array[wp.vec3],surface:wp.array[wp.vec3],current:wp.array[wp.vec3],phase:wp.array[int],recovery:wp.array[int],source_error:wp.array[int]):
    i=wp.tid();scale=float(1.0)
    if phase[0]==3:scale=2.0
    if i<points.shape[0]:
        points[i]=base[i]*scale
        if phase[0]==2:points[i]=wp.vec3(wp.nan)
    if i<current.shape[0]:
        current[i]=surface[i]*scale
        if phase[0]==5 and i==0:current[i]=wp.vec3(wp.nan)
    if i==0:
        recovery[0]=int(phase[0]!=0 and phase[0]!=7)
        source_error[0]=0
        if phase[0]==4:source_error[0]=64

@wp.kernel
def consume(phase:wp.array[int],recovery:wp.array[int],error:wp.array[int],count:wp.array[int],minimum:wp.array[wp.float64],keep:wp.array[int],quality:wp.array[wp.float64],input_count:wp.array[int],ref_error:wp.array[int],ref_valid:wp.array[int],ref_count:wp.array[int],ref_minimum:wp.array[wp.float64],ref_keep:wp.array[int],ref_quality:wp.array[wp.float64],private_rms:wp.array[wp.float64],public_rms:wp.array[wp.float64],meta:wp.array2d[int],metrics:wp.array2d[wp.float64],bad:wp.array[int]):
    i=wp.tid();row=phase[0];expected=ref_error[0]
    if recovery[0]!=0 and ref_valid[0]==0:expected|=16
    if i==0:
        meta[row,0]=error[0];meta[row,1]=expected;meta[row,2]=count[0];meta[row,3]=ref_count[0];meta[row,4]=ref_valid[0];meta[row,5]=recovery[0]
        metrics[row,0]=minimum[0];metrics[row,1]=ref_minimum[0];metrics[row,2]=private_rms[0];metrics[row,3]=public_rms[0]
        if error[0]!=expected:wp.atomic_or(bad,0,1)
        minimum_bad=minimum[0]!=ref_minimum[0]
        if recovery[0]==0:minimum_bad=wp.abs(minimum[0]-ref_minimum[0])>wp.float64(1e-12)
        if error[0]==0 and (count[0]!=ref_count[0] or minimum_bad):wp.atomic_or(bad,0,2)
        if recovery[0]!=0 and private_rms[0]!=wp.float64(0.0):wp.atomic_or(bad,0,4)
    if error[0]==0:
        if keep[i]!=ref_keep[i]:wp.atomic_or(bad,0,8)
        if i<input_count[0]:
            quality_bad=quality[i]!=ref_quality[i]
            if recovery[0]==0:quality_bad=wp.abs(quality[i]-ref_quality[i])>wp.float64(1e-12)
            if quality_bad:wp.atomic_or(bad,0,16)

@wp.kernel
def advance(phase:wp.array[int],condition:wp.array[int]):
    phase[0]+=1;condition[0]=int(phase[0]<8)

p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');p.add_argument('--dot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
c,s=recovered_selection();e=s.recovery_extraction;device=c.device
base=wp.clone(c.points);surface=wp.clone(c.domain.points)
phase=wp.zeros(1,dtype=int,device=device);condition=wp.zeros(1,dtype=int,device=device);bad=wp.zeros(1,dtype=int,device=device)
meta=wp.zeros((8,6),dtype=int,device=device);metrics=wp.zeros((8,4),dtype=wp.float64,device=device)
ordinary=warpower.ExtractionWorkspace(c.domain,c.points,c.full_tetrahedra,c.full_tetrahedra_count,source_error=c.mesh.error)
recovered=warpower.ExtractionWorkspace(c.domain,c.points,c.full_tetrahedra,c.full_tetrahedra_count,source_error=c.mesh.error,containment_recovery=True)
wp.load_module(module=__name__,device=device)

def compare(reference):
    reference.run()
    wp.launch(consume,len(s.keep),inputs=[phase,c.material_recovery,s.error,s.retained_count,s.minimum,s.keep,s.quality,c.full_tetrahedra_count,reference.error,reference.valid,reference.tetrahedra_count,reference.minimum_radius_ratio,reference.selected,reference._input_quality,e.surface_rms_distance,reference.surface_rms_distance,meta,metrics,bad],device=device)
def step():
    wp.launch(produce,max(len(base),len(surface)),inputs=[base,c.points,surface,c.domain.points,phase,c.material_recovery,c.mesh.error],device=device)
    c.domain.refit();s.evaluate()
    wp.capture_if(c.material_recovery,lambda:compare(recovered),lambda:compare(ordinary))
    wp.launch(advance,1,inputs=[phase,condition],device=device)
with wp.ScopedCapture() as cap:
    phase.zero_();condition.fill_(1);bad.zero_();wp.capture_while(condition,step)
wp.synchronize();wp.capture_debug_dot_print(cap.graph,str(args.dot),verbose=True)
if args.profile:
    import torch
    cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(cap.graph);wp.synchronize()
if args.profile:cuda.cudaProfilerStop()
a=meta.numpy();m=metrics.numpy()
r=dict(scope='First native launch: eight captured producer/domain-refit/selection/full-reference/consumer phases. Ordinary/recovered selection, scaled domain/sites, nonfinite sites/domain, source failure and recovery. Public full extraction runs alongside as a device oracle; its diagnostic outputs remain available. Setup and final readback excluded.',phases=['ordinary','recovered','invalid_sites','scaled_recovered','source_failure','invalid_domain','recovery','ordinary_again'],columns=['selection_error','full_reference_normalized_error','selected_count','reference_count','reference_valid','material_recovery'],metadata=a.tolist(),metric_columns=['selection_minimum','reference_minimum','private_rms_unavailable_zero','public_rms'],metrics=m.tolist(),device_comparison_error=int(bad.numpy()[0]),phase=int(phase.numpy()[0]),quality_comparison='Repaired selection uses exact equality to identical full geometry kernels; unchanged ordinary retained classifier uses absolute 1e-12 against the different full classifier.')
args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

assert r['phase']==8 and r['device_comparison_error']==0
assert a[:,2].tolist()==[3,5,0,5,0,0,5,3]
assert a[:,0].tolist()==[0,0,20,0,18,17,0,0]
