"""Plot every measured torus round budget; data are complete fresh solves."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=Path('.work/vtm');source=w/'local-backtracking-final-independent-v2.json';d=json.loads(source.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'warpower-local-backtracking-v1'})
fig,axes=plt.subplots(1,2,figsize=(10.4,4.5));records=[]
for phase,label,color,marker in [('before','Default','#245e91','o'),('enabled','Local retries: 12','#ca5a22','s')]:
 rows=[]
 for rounds,name in [(1,'torus-round1'),(2,'torus-round2'),(4,'torus1000')]:
  r=d['cases'][name][phase];q=r['quality'];rows.append(dict(policy=label,rounds=rounds,milliseconds=r['median_ms'],minimum_ratio=q['radius_ratio_percentiles']['0'],rms_percent_diagonal=q['two_sided_area_weighted_rms_percent_bbox_diagonal']))
 records+=rows
 for ax,key in zip(axes,['minimum_ratio','rms_percent_diagonal']):
  ax.plot([r['milliseconds'] for r in rows],[r[key] for r in rows],marker=marker,color=color,label=label,lw=1.7,ms=6)
  for r in rows:
   ax.annotate(str(r['rounds']), (r['milliseconds'],r[key]),xytext=(5,6 if phase=='enabled' else -13),textcoords='offset points',color=color,fontsize=9)
  ax.grid(alpha=.16);ax.set_xlabel('Complete replay time (ms)');ax.set_xlim(295,760)
axes[0].set_ylabel('Minimum radius ratio (higher is better)');axes[1].set_ylabel('Two-sided sampled RMS, % of box diagonal')
axes[0].legend(frameon=False,loc='upper left');axes[0].set_ylim(.055,.175)
fig.suptitle('Torus, 1,000 sites · complete captured solve',x=.08,ha='left',fontsize=15,fontweight='bold')
fig.text(.08,.88,'L40 · Warp 1.19 · seed 2026 · median of 3 replays · point labels = quality rounds',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.025,1,.86]);stem=w/'local-backtracking-quality-frontier-v1'
assert not stem.with_suffix('.png').exists();fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
stem.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Complete fresh runs, all measured1/2/4-round budgets; sampled RMS is independently measured, not a Hausdorff bound.',records=records),indent=2)+'\n')
