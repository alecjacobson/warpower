"""Private #169 hook; no production/ordinary seed checker is changed."""
import warp as wp
from warpower._prepared_seeds import PreparedSeedQueries,prepare
from seed_stop_filter import check_stopping

def install():
    for block in [32,256]:
        wp.load_module(module='seed_stop_filter',device='cuda:0',block_dim=block)
    def run(self):
        wp.launch(prepare,self.capacity,inputs=self._prepare_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(check_stopping,32*self.capacity,inputs=self._check_inputs,block_dim=self.block_dim,device=self.device)
    PreparedSeedQueries.run=run
