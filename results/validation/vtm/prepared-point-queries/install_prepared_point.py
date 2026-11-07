"""Private #170 hook, selecting one prepared-query arithmetic variant."""
import warp as wp
from warpower._prepared_seeds import PreparedSeedQueries,prepare
from prepared_point_queries import check_point,check_analytic_first

def install(mode):
    kernel={'point':check_point,'analytic':check_analytic_first}[mode]
    for block in [32,256]:
        wp.load_module(module='prepared_point_queries',device='cuda:0',block_dim=block)
    def run(self):
        wp.launch(prepare,self.capacity,inputs=self._prepare_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(kernel,32*self.capacity,inputs=self._check_inputs,block_dim=self.block_dim,device=self.device)
    PreparedSeedQueries.run=run
