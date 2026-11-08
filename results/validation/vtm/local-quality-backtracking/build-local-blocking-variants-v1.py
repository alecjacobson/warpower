"""Keep versioned direct and old-neighborhood blocking pilots separate."""
from pathlib import Path
w=Path('.work/vtm');s=(w/'local_backtracking_v1.py').read_text()
v2=s.replace('factor[i]*=wp.float64(.5)','factor[i]=wp.float64(0)')
v2=v2.replace('"""Private bounded cell-local quality retries, with original global fallback."""','"""Private direct-vertex partial rollback with original global fallback."""')
(w/'local_backtracking_v2.py').write_text(v2)
# Old-star expansion is a retry heuristic, not a geometric certificate. Every
# trial still rebuilds and certifies complete current input before acceptance.
v3=v2.replace('core._local_marks=wp.zeros','core._local_seeds=wp.zeros(core.n,dtype=int,device=core.device)\n    core._local_marks=wp.zeros')
start=v3.index('@wp.kernel\ndef mark(');end=v3.index('@wp.kernel\ndef shrink',start)
v3=v3[:start]+'''@wp.kernel
def seed_bad(tets:wp.array2d[int],count:wp.array[int],keep:wp.array[int],quality:wp.array[wp.float64],
             previous:wp.array[wp.float64],seeds:wp.array[int]):
    i=wp.tid()
    if i<count[0] and keep[i]!=0 and quality[i]<previous[0]*(wp.float64(1)-wp.float64(1e-10)):
        for k in range(4):wp.atomic_or(seeds,tets[i,k],1)

@wp.kernel
def mark_old_star(tets:wp.array2d[int],count:wp.array[int],seeds:wp.array[int],
                  original:wp.array[wp.vec3],points:wp.array[wp.vec3],mask:wp.array[int],
                  marks:wp.array[int],any_mark:wp.array[int]):
    i=wp.tid()
    if i<count[0]:
        adjacent=int(0)
        for k in range(4):adjacent|=seeds[tets[i,k]]
        if adjacent!=0:
            for k in range(4):
                v=tets[i,k]
                if mask[v]!=0 and wp.length_sq(wp.vec3d(points[v])-wp.vec3d(original[v]))>wp.float64(0):
                    wp.atomic_or(marks,v,1);wp.atomic_or(any_mark,0,1)

''' + v3[end:]
start=v3.index('        c=core;s=c.selection;c._local_marks.zero_()');end=v3.index('        wp.copy(c._local_ready,c._local_any)',start)
v3=v3[:start]+'''        c=core;s=c.selection;c._local_marks.zero_();c._local_any.zero_();c._local_seeds.zero_()
        launch(seed_bad,len(c.full_tetrahedra),[c.full_tetrahedra,c.full_tetrahedra_count,s.keep,s.quality,
               s.previous_minimum,c._local_seeds])
        launch(mark_old_star,len(c._old_tets),[c._old_tets,c._old_count,c._local_seeds,c._original,c.points,
               s.mask,c._local_marks,c._local_any])
''' + v3[end:]
v3=v3.replace('Private direct-vertex partial rollback','Private old-neighborhood partial rollback')
(w/'local_backtracking_v3.py').write_text(v3)
for version in [2,3]:
 s=(w/'local_backtracking_full_v1.py').read_text().replace('local_backtracking_v1',f'local_backtracking_v{version}').replace("'halved_sites'","'blocked_sites'")
 (w/f'local_backtracking_full_v{version}.py').write_text(s)
