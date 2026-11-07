"""Private strict-point hook; exhaustive and rejected-cell-stop variants."""
import warp as wp
from warpower._prepared_seeds import PreparedSeedQueries, prepare
from prepared_strict_queries import check_strict


def install(stop):
    for block in [32, 256]:
        wp.load_module(module='prepared_strict_queries', device='cuda:0', block_dim=block)
    def run(self):
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(check_strict, 32*self.capacity, inputs=[*self._check_inputs, stop], block_dim=self.block_dim, device=self.device)
    PreparedSeedQueries.run = run
