"""Diagnostic counters only; extra launches exclude these runs from timing claims."""
import json
import os
from pathlib import Path
import warp as wp
from prepared_split_queries import SplitPreparedSeedQueries

wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def record(count:wp.array[int],live:wp.array[int],pending:wp.array[wp.uint32],stats:wp.array[wp.int64]):
    i=wp.tid()
    if i==0:
        wp.atomic_add(stats,0,wp.int64(1))
    if i>=count[0]:
        return
    value=pending[i]
    wp.atomic_add(stats,1,wp.int64(1))
    wp.atomic_add(stats,2,wp.int64(32))
    if value!=wp.uint32(0):
        wp.atomic_add(stats,3,wp.int64(1))
        wp.atomic_add(stats,4,wp.int64(wp.bit_count(value)))
        if value==wp.uint32(0xFFFFFFFF):
            wp.atomic_add(stats,5,wp.int64(1))
        if live[i]==0:
            wp.atomic_add(stats,7,wp.int64(1))
    if live[i]!=0:
        wp.atomic_add(stats,6,wp.int64(1))

instances=[]
class Profiled(SplitPreparedSeedQueries):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stats=wp.zeros(8,dtype=wp.int64,device=self.device)
        wp.load_module(module=__name__,device=self.device,block_dim=32)
        instances.append(self)
    def run(self):
        super().run()
        wp.launch(record,self.capacity,inputs=[self._check_inputs[2],self._check_inputs[4],self.pending,self.stats],block_dim=32,device=self.device)

import warpower._prepared_seeds as module
module.PreparedSeedQueries=Profiled
from benchmarks.mesher import main
main()
names=['calls','cells','query_lanes','fallback_cells','fallback_lanes','all_lane_fallback_cells','accepted_cells','fallback_rejected_cells']
rows=[dict(site_capacity=len(x._check_inputs[0]),cell_capacity=x.capacity,counts=dict(zip(names,map(int,x.stats.numpy())))) for x in instances]
output=Path(os.environ['WARPOWER_PREPARED_SPLIT_DIAGNOSTIC'])
output.write_text(json.dumps(dict(scope='Instrumented complete solves; counters include every graph execution in the benchmark. Extra diagnostic kernels; timings are not performance evidence.',instances=rows),indent=2)+'\n')
