"""Standalone scientific figure from independently checked schedule experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path.cwd();out=root/'results/validation/vtm/quality-frontier'
rows=[]
for part in ['exploration','polish']:
    rows.extend(json.loads((out/part/'results.json').read_text()))
assert len(rows)==12 and all(r['valid']==1 and r['quota_clean'] and 'independent' in r for r in rows)
by_name={r['name']:r for r in rows}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(11.6,7.5),layout='constrained')
metrics=[('Minimum radius ratio',lambda q:q['radius_ratio_percentiles']['0']),('1st-percentile radius ratio',lambda q:q['radius_ratio_percentiles']['1']),('Median radius ratio',lambda q:q['radius_ratio_percentiles']['50']),('Two-sided boundary RMS (% bbox diagonal)',lambda q:q['two_sided_area_weighted_rms_percent_bbox_diagonal'])]
for ax,(title,metric) in zip(axes.ravel(),metrics):
    ax.set_title(title,loc='left',fontweight='medium',pad=10)
    ax.set_xlabel('Complete resident solve (ms)')
    ax.set_axisbelow(True);ax.grid(color='#e6e9ed',lw=.7)
    for row in rows:
        name=row['name'];fixed=name.startswith('fixed')
        color='#286e96' if fixed else '#be7947'
        marker='D' if fixed else 'o';size=42
        if name=='control':color='#192c3c';marker='*';size=150
        if name=='fixed-32':color='#00886d';marker='D';size=95
        ax.scatter(row['median_ms'],metric(row['independent']['quality']),c=color,marker=marker,s=size,linewidth=.6,edgecolor='white',zorder=4)
    for name,label,offset in [('control','Default',(-8,10)),('fixed-32','Fixed 32',(9,-14)),('fixed-28','Fixed 28',(8,9))]:
        row=by_name[name]
        ax.annotate(label,(row['median_ms'],metric(row['independent']['quality'])),xytext=offset,textcoords='offset points',fontsize=9,ha='right' if name=='control' else 'left',color='#192c3c')
    ax.margins(x=.13,y=.22)
    if title.startswith(('Minimum','1st-percentile')):ax.set_ylim(bottom=0)
    if title.startswith('Two-sided'):ax.text(.02,.92,'Lower is better',transform=ax.transAxes,color='#617080',fontsize=9)
    else:ax.text(.02,.92,'Higher is better',transform=ax.transAxes,color='#617080',fontsize=9)
handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#be7947',markeredgecolor='none',label='Progressive schedules'),Line2D([],[],marker='D',color='none',markerfacecolor='#286e96',markeredgecolor='none',label='Fixed-site schedules'),Line2D([],[],marker='*',markersize=12,color='none',markerfacecolor='#192c3c',markeredgecolor='none',label='Progressive default'),Line2D([],[],marker='D',color='none',markerfacecolor='#00886d',markeredgecolor='none',label='Fixed 32 candidate')]
fig.legend(handles=handles,ncols=4,loc='outside lower center',frameon=False,fontsize=9)
fig.suptitle('VTM schedule study: time and measured mesh quality\n1,000 sites · identical torus surface · seed 2026 · NVIDIA L40 · three fresh replays',fontsize=15,fontweight='medium')
for extension in ['png','pdf']:
    fig.savefig(out/f'quality-time.{extension}',dpi=180,bbox_inches='tight')
print('Saved quality-time.png and quality-time.pdf')
