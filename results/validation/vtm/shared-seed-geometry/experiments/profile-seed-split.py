"""Diagnostic unresolved-cell fractions; additional atomics invalidate timing claims."""
import json
from pathlib import Path
import warp as wp
import warpower._reuse_delaunay as reuse
from warpower._mesher import VariationalMeshingWorkspace
from seed_filter_split import SplitSeeds

wp.set_module_options({'enable_backward':False})


@wp.kernel
def count_states(count:wp.array[int], pending:wp.array[int], live:wp.array[int], stats:wp.array[int]):
    i=wp.tid()
    if i<count[0]:
        wp.atomic_add(stats,0,1)
        if pending[i]!=0:
            wp.atomic_add(stats,1,1)
            if live[i]!=0:
                wp.atomic_add(stats,2,1)
        elif live[i]!=0:
            wp.atomic_add(stats,3,1)


instances=[]


class MeasuredSeeds(SplitSeeds):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stats=wp.zeros(4,dtype=int,device=self.device)
        wp.load_module(module=__name__,device=self.device)
        instances.append(self)

    def run(self):
        super().run()
        wp.launch(count_states,len(self.unique),inputs=[self.unique_count,self.pending,self.live,self.stats],device=self.device)


original=VariationalMeshingWorkspace._solve
def solve(self):
    for instance in instances:instance.stats.zero_()
    original(self)

VariationalMeshingWorkspace._solve=solve
reuse.CertifiedSeeds=MeasuredSeeds
from benchmarks.mesher import main
main()
rows=[dict(capacity=len(i.source),total=int(i.stats.numpy()[0]),unresolved=int(i.stats.numpy()[1]),unresolved_retained=int(i.stats.numpy()[2]),bounds_only_retained=int(i.stats.numpy()[3])) for i in instances]
Path('.work/vtm/seed-split-fractions.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
