"""Link the qualified hierarchy proof report without rewriting historical results."""
from pathlib import Path
root=Path.cwd();assert (root/'docs/exterior-proof-hierarchy.md').exists()
p=root/'docs/proven-exterior-recovery.md';s=p.read_text();needle='## Whole-cell proof\n';assert s.count(needle)==1
s=s.replace(needle,'''Current traversal uses [conservative surface-hierarchy bounds](exterior-proof-hierarchy.md)
for both plane candidates and exact rational-centroid rays. The report distinguishes
safe removal of disjoint coplanar-ray ambiguity from exact leaf predicate decisions
and records the newer performance/capture/memory qualification. The measurements
below remain the original recovery-policy study.

'''+needle);p.write_text(s)
p=root/'docs/architecture.md';s=p.read_text();needle='bounds filter the exact integer signs. All selection, rollback, and subsequent\n';assert s.count(needle)==1
s=s.replace(needle,'''bounds filter the exact integer signs. [Hierarchy exclusion](exterior-proof-hierarchy.md)
uses current exact surface boxes and conservative rational-centroid slab bounds
to avoid unrelated triangles without rounding the centroid. All selection, rollback, and subsequent
''');p.write_text(s)
