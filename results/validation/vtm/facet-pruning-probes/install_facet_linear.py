"""Private #164 opt-in. All source and default dispatch remain untouched."""
import warp as wp
from warpower._proved_completion import SeededFacetCompletion
from facet_linear_box_search import search_linear_box as search_tree

def install(min_sites=0):
    for block in [32,256]:
        wp.load_module(module='facet_linear_box_search',device='cuda:0',block_dim=block)
    original=SeededFacetCompletion._search_faces
    def search(self,tree):
        if len(self.points)<min_sites:
            return original(self,tree)
        c=self._certificate
        self._launch(search_tree,4*self.capacity*self._workers,[
            self.points,tree,self.tetrahedra,c._boundary,c._boundary_count,
            self._candidates,self._workers,self.error,c._face_supported,c.face_statistics
        ],block_dim=self._query_block_dim)
    SeededFacetCompletion._search_faces=search
