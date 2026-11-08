"""Standalone current-runtime schedule frontier; no CPU speedup inferred."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

w=Path('.work/vtm')
paths=[w/'fixed-site-current-pilot-v1-independent.json',w/'fixed-site-current-larger-v1-independent.json']
reports=[json.loads(p.read_text()) for p in paths]
assert all(d['complete'] for d in reports)
rows={k:v for d in reports for k,v in d['cases'].items() if 'independent' in v}
fig,axes=plt.subplots(1,3,figsize=(12.5,3.9),layout='constrained')
colors={'progressive':'#315d9b','fixed32':'#c75f30'}
for schedule,label in [('progressive','Progressive'),('fixed32','Fixed sites · 32 steps')]:
    ns=[n for n in [1000,4096,8192,16384] if f'torus{n}-{schedule}' in rows]
    data=[rows[f'torus{n}-{schedule}'] for n in ns]
    series=[[r['median_ms']/1000 for r in data],[r['independent']['quality']['radius_ratio_percentiles']['1'] for r in data],[r['independent']['quality']['two_sided_area_weighted_rms_percent_bbox_diagonal'] for r in data]]
    for axis,y in zip(axes,series):
        axis.plot(ns,y,'o-',color=colors[schedule],lw=2,ms=5,label=label)
for axis in axes:
    axis.set_xscale('log',base=2)
    axis.set_xticks([1000,4096,8192,16384],['1k','4k','8k','16k'])
    axis.set_xlabel('Requested sites')
    axis.spines[['top','right']].set_visible(False)
    axis.grid(alpha=.18)
axes[0].set_yscale('log')
axes[0].set_ylabel('Complete resident solve (s) ↓')
axes[1].set_ylabel('1st-percentile radius ratio ↑')
axes[2].set_ylabel('Two-sided boundary RMS (% bbox) ↓')
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Existing schedules on the current Warp runtime',fontsize=14)
for suffix in ['png','pdf','svg']:
    path=w/f'fixed-site-current-frontier-v1.{suffix}'
    assert not path.exists()
    fig.savefig(path,dpi=180)
plt.close(fig)
(w/'fixed-site-current-plot-source-v1.json').write_text(json.dumps(dict(scope=__doc__,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
