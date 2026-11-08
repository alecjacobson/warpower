"""Complete source metadata in the not-yet-executed production grid harness."""
from pathlib import Path
p=Path('.work/vtm/bench-larger-prefix-production-grid-v1.py');s=p.read_text();old='["warpower/_proved_completion.py", "warpower/_unproved_queries.py"]';assert s.count(old)==1
s=s.replace(old,'["warpower/_active_delaunay.py", "warpower/_vtm.py", "warpower/_proved_completion.py", "warpower/_unproved_queries.py"]');p.write_text(s)
