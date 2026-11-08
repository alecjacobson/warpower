"""Promote the simpler measured kernels, with explicit borrowed scratch ownership."""
from pathlib import Path
import ast
w=Path('.work/vtm')
p=Path('warpower/_unproved_queries.py');assert not p.exists();p.write_text((w/'proposed_unproved_queries_v1.py').read_text())
p=Path('warpower/_proved_completion.py');s=p.read_text();needle='''            self.face_statistics = wp.zeros(2, dtype=int)
''';assert s.count(needle)==1;s=s.replace(needle,needle+'''            self._prepared_unproved = None
''')
needle='''    def _global_spheres(self, tree):
''';assert s.count(needle)==1;s=s.replace(needle,needle+'''        if self._prepared_unproved is not None and self.point_parallel:
            self._prepared_unproved.run(tree)
            return
''')
needle='''            seed_filter.tree,
            **kwargs,
        )

    def run(self):''';assert s.count(needle)==1
replacement='''            seed_filter.tree,
            **kwargs,
        )
        seed = seed_filter._prepared
        if seed is not None and 4096 < len(seed_filter.points) <= 16384:
            from ._stackless_seeds import StacklessSeedQueries
            from ._unproved_queries import PreparedUnprovedQueries

            c = self._certificate
            if (
                isinstance(seed, StacklessSeedQueries)
                and len(c.tetrahedra) <= len(seed.shapes)
                and len(c.tetrahedra) <= len(seed.boxes)
            ):
                # Seed queries finish before completion begins. Their geometry
                # is dead scratch; every later consumer prepares its own current
                # cells again. Keep direct array references, without a cycle.
                c._prepared_unproved = PreparedUnprovedQueries(
                    c.points,
                    c.tetrahedra,
                    c.tetrahedra_count,
                    c.tree,
                    c.error,
                    c.statistics,
                    c._proved,
                    c.proof_statistics,
                    segment=seed.segment,
                    shapes=seed.shapes,
                    boxes=seed.boxes,
                )

    def run(self):'''
s=s.replace(needle,replacement);p.write_text(s)
p=Path('tests/test_unproved_queries.py');assert not p.exists();s=(w/'test_prepared_new_cell_v1.py').read_text().replace('from prepared_new_cell_v1 import PreparedUnprovedQueries, check_prepared_unproved, prepare_unproved','from warpower._unproved_queries import (\n    PreparedUnprovedQueries, check_prepared_unproved, prepare_unproved,\n)')
s=s.replace('''        prepared = PreparedUnprovedQueries(*original, segment=hierarchy._segment, block_dim=block)''','''        prepared = PreparedUnprovedQueries(
            *original,
            segment=hierarchy._segment,
            shapes=wp.empty(capacity, dtype=Sphere),
            boxes=wp.empty(capacity, dtype=sb.Box),
            block_dim=block,
        )''')
p.write_text(s)
old=ast.parse((w/'prepared_new_cell_v1.py').read_text());new=ast.parse(Path('warpower/_unproved_queries.py').read_text())
for name in ['prepare_unproved','check_prepared_unproved']:
 a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);assert ast.dump(a)==ast.dump(b),name
print('Simple v1 kernels unchanged; borrowed geometry enabled only for the existing private larger route')
