"""Prepare a reviewable source draft only; no repository mutation or execution."""
from pathlib import Path
import ast,json,hashlib
w=Path('.work/vtm');s=(w/'prepared_new_cell_v1.py').read_text()
s=s.replace('''"""Private #233 primitive: unchanged predicates/counters, shared current geometry.

The caller supplies finite original sites, canonical owners, valid cell indices,
and the rebuilt segment/permutation, as required by private facet completion.
No production installation or scratch sharing is provided in this first probe.
"""''','''"""Share current-cell sphere geometry and traverse current original-site bounds.

Private completion stage: finite sites, canonical owners, valid cell indices and
the rebuilt current segment are required. Scratch is borrowed from the seed
query stage after it finishes; each invocation prepares current geometry anew.
"""''')
s=s.replace('''        segment,
        block_dim=32,''','''        segment,
        shapes,
        boxes,
        block_dim=32,''')
s=s.replace('''        self.shapes = wp.empty(self.capacity, dtype=Sphere, device=self.device)
        self.boxes = wp.empty(self.capacity, dtype=sb.Box, device=self.device)''','''        if len(shapes) < self.capacity or len(boxes) < self.capacity:
            raise ValueError("Borrowed sphere geometry must cover every completion cell slot")
        if shapes.device != self.device or boxes.device != self.device:
            raise ValueError("Borrowed sphere geometry must share the query device")
        self.shapes, self.boxes, self.segment = shapes, boxes, segment''')
s=s.replace('''    def run(self):
        wp.launch(''','''    def run(self, tree=None):
        current = self.query_inputs[3] if tree is None else tree
        if current.ids.ptr != self.segment.ids.ptr or current.n != self.segment.n:
            raise ValueError("Completion query tree changed its segment association")
        wp.launch(''')
s=s.replace('''            inputs=self.query_inputs,''','''            inputs=[*self.query_inputs[:3], current, *self.query_inputs[4:]],''')
p=w/'proposed_unproved_queries_v1.py';assert not p.exists();p.write_text(s)
old=ast.parse((w/'prepared_new_cell_v1.py').read_text());new=ast.parse(s)
for name in ['prepare_unproved','check_prepared_unproved']:
 a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);assert ast.dump(a)==ast.dump(b),name
print('Draft kernels unchanged; helper has explicit borrowed buffers and association/capacity guards')
