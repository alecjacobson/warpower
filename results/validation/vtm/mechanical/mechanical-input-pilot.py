"""Procedural counterbored mounting plate input, before device meshing."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from examples.vtm_inputs import write_off
w=Path('.work/vtm/mechanical-input');w.mkdir(exist_ok=True)
hx,hy,zlo,zstep,zhi,rbore,rcounter=1.25,.85,-.35,.05,.35,.35,.65
angles=[(2*np.pi*i/32,None) for i in range(32)]
for x,y in [(hx,hy),(-hx,hy),(-hx,-hy),(hx,-hy)]:angles.append((np.arctan2(y,x)%(2*np.pi),(x,y)))
angles.sort(key=lambda v:v[0]);theta=np.array([v[0] for v in angles]);n=len(theta);assert n==36
outer=[]
for angle,corner in angles:
 c,s=np.cos(angle),np.sin(angle)
 if corner:outer.append(corner)
 elif abs(c)/hx>=abs(s)/hy:outer.append((np.copysign(hx,c),hx*s/abs(c)))
 else:outer.append((hy*c/abs(s),np.copysign(hy,s)))
outer=np.array(outer);circle=np.column_stack((np.cos(theta),np.sin(theta)))
ring=lambda xy,z:np.column_stack((xy,np.full(n,z)))
# Ob, Ot, Ib, lower shoulder, upper shoulder, It.
p=np.vstack([ring(outer,zlo),ring(outer,zhi),ring(rbore*circle,zlo),ring(rbore*circle,zstep),ring(rcounter*circle,zstep),ring(rcounter*circle,zhi)]).astype(np.float32)
faces=[]
def quad(a,b,c,d):faces.extend([(a,b,c),(a,c,d)])
for i in range(n):
 j=(i+1)%n
 ob,ot,ib,sl,su,it=[k*n+i for k in range(6)];obn,otn,ibn,sln,sun,itn=[k*n+j for k in range(6)]
 quad(ob,obn,otn,ot) # outer wall, outward
 quad(ob,ib,ibn,obn) # bottom cap, downward
 quad(ot,otn,itn,it) # top cap, upward
 quad(ib,sl,sln,ibn) # lower bore, inward
 quad(su,sun,sln,sl) # shoulder, upward
 quad(su,it,itn,sun) # upper bore, inward
f=np.array(faces,np.int32);tri=p[f].astype(float);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);assert np.all(np.linalg.norm(normal,axis=1)>0)
edges={}
for a,b,c in f:
 for x,y in [(a,b),(b,c),(c,a)]:edges.setdefault(tuple(sorted((int(x),int(y)))),[]).append((int(x),int(y)))
assert all(len(es)==2 and es[0]==es[1][::-1] for es in edges.values())
assert len(p)-len(edges)+len(f)==0
volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
area=lambda xy:abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))/2
# Integrate the actual float32 polygonal bore, not an ideal circular cylinder.
outer_area=area(p[:n,:2].astype(float));lower_area=area(p[2*n:3*n,:2].astype(float));upper_area=area(p[4*n:5*n,:2].astype(float))
expected=(float(p[n,2])-float(p[0,2]))*outer_area-(float(p[3*n,2])-float(p[2*n,2]))*lower_area-(float(p[5*n,2])-float(p[4*n,2]))*upper_area
assert np.isclose(volume,expected,rtol=1e-13,atol=1e-14)
off=w/'counterbored-plate.off';write_off(off,p,f)
check=subprocess.run(['.work/vtm/surface-reference',str(off)],text=True,capture_output=True,check=True);assert int(check.stdout.split()[2])==0,check.stdout
(w/'self-intersections.txt').write_text(check.stdout)
result=dict(scope='Procedural input validation only; no device validation or meshing yet. Analytic mechanical substitute, not an Alliez source asset.',vertices=len(p),triangles=len(f),edges=len(edges),boundary_euler=len(p)-len(edges)+len(f),oppositely_paired_edges=True,nonzero_triangle_areas=True,volume=volume,independent_cross_section_volume=expected,self_intersection_pairs=0,off_sha256=hashlib.sha256(off.read_bytes()).hexdigest(),parameters=dict(half_width=hx,half_height=hy,bottom=zlo,shoulder=zstep,top=zhi,through_radius=rbore,counterbore_radius=rcounter,uniform_azimuths=32,extra_exact_corner_rays=4),surface_checker_binary_sha256=hashlib.sha256(Path('.work/vtm/surface-reference').read_bytes()).hexdigest())
(w/'provenance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
