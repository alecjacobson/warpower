"""Attach the completed cross-reservation comparison to unexecuted report/publication drivers."""
from pathlib import Path
w=Path('.work/vtm');p=w/'write-larger-prefix-report-v1.py';s=p.read_text()
needle="s=f'''# Larger device-selected active prefixes";assert needle in s
s=s.replace(needle,"cross=read('larger-prefix-current-reservation-comparison-v1.json');assert cross['complete']\nx4=cross['cases']['reserved16384-target4096'];x8=cross['cases']['reserved16384-target8192']\n"+needle)
needle='''JSON; reduced constructor capacity does not mean fewer optimizer iterations.
''';assert needle in s
s=s.replace(needle,needle+'''
A [cross-reservation check](../{out}/larger-prefix-current-reservation-comparison-v1.json)
also finds identical geometry and work across the saved post-change reservations.
Using the primary medians, a 4K target reserved at16K records
{x4['reserved_over_dedicated']:.3f}× its dedicated4K time; the 8K target reserved
at16K records {x8['reserved_over_dedicated']:.3f}× its dedicated8K time. These reuse
the existing samples; no separate interleaved cross-reservation timing experiment
was performed.
'''.replace('at16K','at 16K').replace('dedicated4K','dedicated 4K').replace('dedicated8K','dedicated 8K'))
p.write_text(s)
p=w/'publish-larger-prefix-v1.py';s=p.read_text();needle="assert read('larger-prefix-pilot-v1-independent.json')['complete']";assert needle in s;s=s.replace(needle,needle+"\nassert read('larger-prefix-current-reservation-comparison-v1.json')['complete']")
s=s.replace("'bench-larger-prefix')","'bench-larger-prefix','compare-larger-prefix')");p.write_text(s)
p=w/'verify-larger-prefix-v1.py';s=p.read_text();needle="assert read('larger-prefix-pilot-v1-independent.json')['complete']";assert needle in s;s=s.replace(needle,needle+"\nassert read('larger-prefix-current-reservation-comparison-v1.json')['complete']");p.write_text(s)
