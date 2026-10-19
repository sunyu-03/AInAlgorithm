"""Layer precedence with concurrent travel, plus a resident-robot comparator.

Deposition endpoints are unchanged. Motion subdivision is for conservative
continuous collision proofs only; CSV deposition beads are never subdivided.
"""
from itertools import combinations
import numpy as np
from shapely import MultiPoint,STRtree
from algorithm.proven_segments import merge_intervals

def route(env,indices,r,current=None,return_home=True):
 current=np.array(env.homes[r] if current is None else current,float)
 rows=[[0.,*current,'W']];trace=[];pending=list(indices)
 while pending:
  choices=[]
  for i in pending:
   t=env.tasks[i]
   for reverse,(a,b) in enumerate(((t.start,t.end),(t.end,t.start))):
    choices.append((float(np.linalg.norm(np.asarray(a)-current)),i,reverse,np.asarray(a),np.asarray(b)))
  _,i,reverse,a,b=min(choices,key=lambda v:(v[0],v[1],v[2]))
  for point,mode,speed in ((a,'T',env.config.process.travel_speed_mm_s),(b,'D',env.config.process.deposition_speed_mm_s)):
   distance=float(np.linalg.norm(point-current))
   if distance>1e-9:
    rows[-1][4]=mode;rows.append([rows[-1][0]+distance/speed,*point,'W']);current=point
  trace.append(dict(task=i,robot_id=r+1,reverse=reverse,work_finish_s=rows[-1][0]));pending.remove(i)
 if return_home:
  d=float(np.linalg.norm(current-env.homes[r]))
  if d>1e-9:rows[-1][4]='T';rows.append([rows[-1][0]+d/env.config.process.travel_speed_mm_s,*env.homes[r],'W'])
 return rows,trace

def parts(rows,base,step):
 tcp=[];arm=[];times=[]
 for a,b in zip(rows,rows[1:]):
  if a[4]=='W':continue
  p=np.array(a[1:3]);q=np.array(b[1:3]);count=max(1,int(np.ceil(np.linalg.norm(q-p)/step)))
  for j in range(count):
   ends=np.array([p+(q-p)*j/count,p+(q-p)*(j+1)/count],np.float32).astype(float)
   tcp.append(MultiPoint(ends).convex_hull);arm.append(MultiPoint(np.vstack((ends,base))).convex_hull)
   times.append((a[0]+(b[0]-a[0])*j/count,a[0]+(b[0]-a[0])*(j+1)/count))
 return tcp,arm,np.asarray(times)

def radii(env,a,b):
 c=env.config.collision
 maximum=max(np.max(np.abs(env.homes[:,:2])),max(abs(v) for t in env.tasks for p in (t.start,t.end) for v in p[:2]))
 eps=max(c.geometry_epsilon_mm,float(8*np.spacing(np.float32(maximum))),1e-5)
 return env.robots[a].tcp_radius_mm+env.robots[b].tcp_radius_mm+eps,env.robots[a].arm_envelope_radius_mm+env.robots[b].arm_envelope_radius_mm+c.arm_clearance_mm+eps

def home_safe(env,r,geometry):
 c=env.config.collision
 for park in range(3):
  if r==park:continue
  tcp,arm=radii(env,r,park);p=np.asarray(env.homes[park][:2],np.float32).astype(float)
  fixed=[MultiPoint([p]).convex_hull,MultiPoint([p,env.bases[park,:2]]).convex_hull]
  for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
   if enabled and any(g.distance(fixed[kind])<=radius for g in geometry[kind]):return False
 return True

def earliest(env,r,new,existing,lower):
 blocked=[];c=env.config.collision
 for other,old in existing.items():
  if other==r or not len(old[2]):continue
  live=np.flatnonzero(old[2][:,1]>=lower-1e-6)
  if not len(live):continue
  old=([old[0][j] for j in live],[old[1][j] for j in live],old[2][live])
  tcp,arm=radii(env,r,other);pairs=set()
  for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
   if enabled:
    found=STRtree(old[kind]).query(new[kind],predicate='dwithin',distance=radius)
    pairs.update(zip(found[0].tolist(),found[1].tolist()))
  blocked.extend((old[2][j,0]-new[2][i,1]-1e-6,old[2][j,1]-new[2][i,0]+1e-6) for i,j in pairs)
 start=max(0.,lower)
 for left,right in merge_intervals(blocked):
  if left<=start<=right:start=right+1e-6
 return start

def eligible(env,indices,r):
 return all(np.linalg.norm(np.asarray(p[:2])-env.bases[r,:2])<=env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for i in indices for p in (env.tasks[i].start,env.tasks[i].end))

def resident(env,step=25.):
 candidates=[]
 layers={}
 for i,t in enumerate(env.tasks):layers.setdefault(t.start[2],[]).append(i)
 for r in range(3):
  if not eligible(env,range(len(env.tasks)),r):continue
  rows=[[0.,*env.homes[r],'W']];trace=[]
  for z,indices in sorted(layers.items()):
   local,tr=route(env,indices,r,rows[-1][1:4],False);offset=rows[-1][0]
   rows[-1][4]=local[0][4];rows.extend([[v[0]+offset,*v[1:]] for v in local[1:]])
   trace.extend(dict(**t,layer_z=z,departure_s=offset) for t in tr)
  d=float(np.linalg.norm(np.asarray(rows[-1][1:4])-env.homes[r]));rows[-1][4]='T';rows.append([rows[-1][0]+d/env.config.process.travel_speed_mm_s,*env.homes[r],'W'])
  geometry=parts(rows,env.bases[r,:2],step)
  if home_safe(env,r,geometry):candidates.append((rows[-1][0],r,rows,trace))
 if not candidates:raise ValueError('No home-safe resident robot')
 end,r,rows,trace=min(candidates,key=lambda x:(x[0],x[1]))
 env.rows=[rows if k==r else [[0.,*h,'W'],[end,*h,'W']] for k,h in enumerate(env.homes)]
 env.index=len(env.tasks);env.makespan=end;env.done=True
 return dict(kind='resident',robot_id=r+1,makespan_s=end,trace=trace)

def pipeline(env,step=25.,cycle=False,grouped=False):
 archive=[[[0.,*h,'W']] for h in env.homes];existing={};barrier=0.;trace=[];diagnostics=[]
 layers={}
 for i,t in enumerate(env.tasks):layers.setdefault(t.start[2],[]).append(i)
 units=[]
 for z,indices in sorted(layers.items()):
  if grouped:
   groups={}
   for i in indices:
    valid=[r for r in range(3) if eligible(env,[i],r)]
    if not valid:raise ValueError(f'Task {i} unreachable')
    mid=(np.asarray(env.tasks[i].start)+env.tasks[i].end)/2
    r=min(valid,key=lambda r:(np.linalg.norm(mid[:2]-env.bases[r,:2]),r));groups.setdefault(r,[]).append(i)
   units.extend((z,g,r) for r,g in sorted(groups.items()))
  else:units.append((z,indices,None))
 for z,indices,assigned in units:
  choices=[]
  for r in range(3):
   if assigned is not None and assigned!=r:continue
   if not eligible(env,indices,r):continue
   local,tr=route(env,indices,r);new=parts(local,env.bases[r,:2],step)
   if not home_safe(env,r,new):continue
   deposits=[(a[0],b[0]) for a,b in zip(local,local[1:]) if a[4]=='D']
   start=earliest(env,r,new,existing,max(archive[r][-1][0],barrier-deposits[0][0]))
   choices.append((start+deposits[-1][1],start+local[-1][0],r,start,local,tr,deposits))
  if not choices:raise ValueError(f'No proven pipeline route at layer {z}')
  desired=len(diagnostics)%3
  pool=[v for v in choices if v[2]==desired] if cycle else choices
  work_end,end,r,start,local,tr,deposits=min(pool or choices,key=lambda x:x[:3])
  if start>archive[r][-1][0]:archive[r].append([start,*env.homes[r],'W'])
  archive[r][-1][4]=local[0][4];archive[r].extend([[row[0]+start,*row[1:]] for row in local[1:]])
  existing[r]=parts(archive[r],env.bases[r,:2],step);barrier=work_end
  trace.extend(dict(**t,layer_z=z,departure_s=start) for t in tr)
  diagnostics.append(dict(z=z,robot_id=r+1,departure_s=start,first_deposition_s=start+deposits[0][0],last_deposition_s=work_end,home_finish_s=end))
  print(f'Pipeline layer {z}: robot={r+1}, work_end={work_end:.3f}, home_end={end:.3f}',flush=True)
 end=max(rows[-1][0] for rows in archive)
 for r in range(3):
  if archive[r][-1][0]<end:archive[r].append([end,*env.homes[r],'W'])
 env.rows=archive;env.index=len(env.tasks);env.makespan=end;env.done=True
 return dict(kind='pipeline_grouped' if grouped else 'pipeline_cycle' if cycle else 'pipeline',makespan_s=end,trace=trace,layers=diagnostics,proof_step_mm=step)
