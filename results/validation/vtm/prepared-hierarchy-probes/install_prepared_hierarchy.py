"""Private #172 hook. Compile before measured complete replay."""
import warp as wp
from warpower._prepared_seeds import PreparedSeedQueries, prepare
from prepared_hierarchy_queries import check_tree


def install(polynomial_nodes):
    for block in [32, 256]:
        wp.load_module(module='prepared_hierarchy_queries', device='cuda:0', block_dim=block)
    def run(self):
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(check_tree, 32*self.capacity, inputs=[*self._check_inputs, polynomial_nodes], block_dim=self.block_dim, device=self.device)
    PreparedSeedQueries.run = run
