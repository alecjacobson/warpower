"""Private #166 opt-in: only Reuse-owned completion's face search changes."""
import warp as wp
from warpower._proved_completion import SeededFacetCompletion
from analytic_face_search import search_analytic

def install():
    for block in [32,256]:
        wp.load_module(module='analytic_face_search',device='cuda:0',block_dim=block)
    def search(self,tree):
        c=self._certificate
        self._launch(search_analytic,4*self.capacity*self._workers,[
            self.points,tree.owners,self.tetrahedra,c._boundary,c._boundary_count,
            self._candidates,self._workers,self.error,c._face_supported,c.face_statistics
        ],block_dim=self._query_block_dim)
    SeededFacetCompletion._search_faces=search
