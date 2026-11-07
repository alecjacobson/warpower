import warp as wp
from warpower._prepared_seeds import PreparedSeedQueries, prepare
from prepared_narrow_tree_queries import check_tree_1, check_tree_8

def install(workers):
    kernel={1:check_tree_1,8:check_tree_8}[workers]
    for block in [32,256]:
        wp.load_module(module='prepared_narrow_tree_queries',device='cuda:0',block_dim=block)
    def run(self):
        wp.launch(prepare,self.capacity,inputs=self._prepare_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(kernel,workers*self.capacity,inputs=self._check_inputs,block_dim=self.block_dim,device=self.device)
    PreparedSeedQueries.run=run
