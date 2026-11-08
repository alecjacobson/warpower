"""Observe local retry budget, acceptance and focal-site factors."""
import warp as wp
wp.set_module_options({'enable_backward':False,'fast_math':False})
@wp.kernel
def observe(mode:wp.array[int],rounds:wp.array[int],iterations:wp.array[int],alpha:wp.array[wp.float64],
            direction:wp.array[wp.vec3d],edge:wp.array[wp.float64],mask:wp.array[int],
            previous:wp.array[wp.float64],minimum:wp.array[wp.float64],accepted:wp.array[int],
            rejections:wp.array[int],error:wp.array[int],points:wp.array[wp.vec3],
            counter:wp.array[int],rows:wp.array2d[wp.float64],used:wp.array[int],factor:wp.array[wp.float64],
            backtracks:wp.array[int],before:wp.array[int]):
    if mode[0]!=0:
        row=counter[0];counter[0]+=1
        if row<rows.shape[0]:
            rows[row,0]=wp.float64(rounds[2]);rows[row,1]=wp.float64(iterations[0]);rows[row,2]=alpha[0]
            rows[row,3]=wp.length(direction[403]);rows[row,4]=edge[403];rows[row,5]=wp.float64(mask[403])
            rows[row,6]=previous[0];rows[row,7]=minimum[0];rows[row,8]=wp.float64(accepted[0])
            rows[row,9]=wp.float64(rejections[0]);rows[row,10]=wp.float64(error[0])
            for axis in range(3):rows[row,11+axis]=wp.float64(points[403][axis])
            rows[row,14]=wp.float64(used[0]);rows[row,15]=factor[403];rows[row,16]=factor[329]
            rows[row,17]=factor[431];rows[row,18]=factor[976];rows[row,19]=wp.float64(backtracks[0]);rows[row,20]=wp.float64(before[0])
