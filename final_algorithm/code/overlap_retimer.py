"""Compress common layer offsets while reserving travel AND stationary waits."""
from itertools import combinations,permutations
from types import SimpleNamespace
import numpy as np
from shapely import MultiPoint,STRtree
from parallel_scheduler import parts,radii
from flexible_parallel import earliest_intervals,GUARD

def reserved_parts(rows,base,home,step):
 tcp,arm,times=parts(rows,base,step);intervals=list(times)
 for a,b in zip(rows,rows[1:]):
  if a[4]!='W' or b[0]<=a[0]:continue
  p=np.asarray(a[1:3],np.float32).astype(float)
  if np.linalg.norm(p-np.asarray(home[:2],np.float32))<1e-5:continue
  tcp.append(MultiPoint([p]).convex_hull);arm.append(MultiPoint([p,base]).convex_hull);intervals.append((a[0],b[0]))
 return tcp,arm,np.asarray(intervals,float).reshape(-1,2)

def pairs(left,right,thresholds,config):
 found=[]
 for kind,enabled in ((0,config.check_tcp_radius),(1,config.check_arm_envelope)):
  if enabled and left[kind] and right[kind]:
   hit=STRtree(right[kind]).query(left[kind],predicate='dwithin',distance=thresholds[kind])
   if hit.shape[1]:found.append(hit.T)
 return np.unique(np.concatenate(found),axis=0) if found else np.empty((0,2),int)

def forbidden(new,old,hit):
 if not len(hit):return np.empty((0,2))
 i,j=hit[:,0],hit[:,1]
 return np.column_stack((old[j,0]-new[i,1]-GUARD,old[j,1]-new[i,0]+GUARD))

def trim_home_padding(rows,home):
 moving=[i for i in range(len(rows)-1) if rows[i][4]!='W']
 if not moving:return None
 rr=[v.copy() for v in rows[moving[0]:moving[-1]+2]]
 if any(np.linalg.norm(np.array(v[1:4])-home)>1e-4 for v in (rr[0],rr[-1])):raise ValueError('Layer does not begin/end at home')
 return rr

def schedule(env,oldrows,metadata,step=10.):
 layers=metadata.get('layers',[]) if metadata else []
 if not layers:
  env.rows=[[v.copy() for v in oldrows[r]] for r in range(3)];env.makespan=max(v[-1][0] for v in env.rows)
  return dict(kind='preserved_no_home_barrier',layers=[],makespan_s=env.makespan,proof_step_mm=step)
 thresholds={(a,b):radii(env,a,b) for a,b in combinations(range(3),2)}
 for (a,b),v in list(thresholds.items()):thresholds[b,a]=v
 archive=[[[0.,*h,'W']] for h in env.homes];existing={};barrier=0.;details=[]
 for layer in layers:
  z=layer['z'];start=layer['layer_start_s'];end=start+layer['duration_s'];local={};geometry={}
  for r,rows in oldrows.items():
   rr=[[v[0]-start,*v[1:]] for v in rows if start-1e-7<=v[0]<=end+1e-7]
   if not rr or abs(rr[0][0])>1e-6 or abs(rr[-1][0]-layer['duration_s'])>1e-6:raise ValueError('Incomplete layer boundary')
   rr=trim_home_padding(rr,env.homes[r])
   if rr:local[r]=rr;geometry[r]=reserved_parts(rr,env.bases[r,:2],env.homes[r],step)
  # Prove moving/work-wait geometry cannot collide with parked robots.
  for r,g in geometry.items():
   for other in range(3):
    if r==other:continue
    p=np.asarray(env.homes[other,:2],np.float32).astype(float);park=([MultiPoint([p]).convex_hull],[MultiPoint([p,env.bases[other,:2]]).convex_hull])
    if len(pairs(g,park,thresholds[r,other],env.config.collision)):raise ValueError('Parked-home proof failed')
  for a,b in combinations(local,2):
   hit=pairs(geometry[a],geometry[b],thresholds[a,b],env.config.collision)
   if len(hit):
    aa=geometry[a][2][hit[:,0]];bb=geometry[b][2][hit[:,1]]
    if np.any(np.minimum(aa[:,1],bb[:,1])-np.maximum(aa[:,0],bb[:,0])>1e-8):raise ValueError('Existing layer fails conservative continuous proof')
  lower=0.;last_d=[]
  for r,rr in local.items():
   deposits=[(a[0],b[0]) for a,b in zip(rr,rr[1:]) if a[4]=='D']
   lower=max(lower,archive[r][-1][0]-rr[0][0],barrier-deposits[0][0]);last_d.append(deposits[-1][1])
  blocked=[]
  # Only parts whose end can overlap any nonnegative-time new segment survive.
  for other,g in list(existing.items()):
   ids=np.flatnonzero(g[2][:,1]>=lower-GUARD);existing[other]=([g[0][i] for i in ids],[g[1][i] for i in ids],g[2][ids])
  for r,new in geometry.items():
   for other,old in existing.items():
    if r==other:continue
    hit=pairs(new,old,thresholds[r,other],env.config.collision)
    if len(hit):blocked.append(forbidden(new[2],old[2],hit))
  departure=earliest_intervals(np.concatenate(blocked) if blocked else [],lower)
  for r,rr in local.items():
   begin=departure+rr[0][0]
   if begin<archive[r][-1][0]-1e-7:raise ValueError('Robot overlaps its own previous route')
   if begin>archive[r][-1][0]+1e-9:
    archive[r][-1][4]='W';archive[r].append([begin,*env.homes[r],'W'])
   archive[r][-1][4]=rr[0][4];archive[r].extend([[v[0]+departure,*v[1:]] for v in rr[1:]])
   g=geometry[r];shifted=(g[0],g[1],g[2]+departure);old=existing.get(r)
   existing[r]=(old[0]+shifted[0],old[1]+shifted[1],np.concatenate((old[2],shifted[2]))) if old else shifted
  barrier=departure+max(last_d);home_end=max(v[-1][0] for v in archive)
  details.append(dict(z=z,original_start_s=start,new_offset_s=departure,last_deposition_s=barrier,home_finish_s=home_end,lower_bound_s=lower))
  print(f'Overlap layer {z}: work {barrier:.3f}s, home {home_end:.3f}s',flush=True)
 end=max(v[-1][0] for v in archive)
 for r in range(3):
  if archive[r][-1][0]<end:
   archive[r][-1][4]='W';archive[r].append([end,*env.homes[r],'W'])
 env.rows=archive;env.makespan=end
 return dict(kind='layer_travel_overlap',layers=details,makespan_s=end,proof_step_mm=step,stationary_waits_reserved=True,within_layer_timing_unchanged=True)

def schedule_independent(env,oldrows,metadata,step=10.):
 layers=metadata.get('layers',[]) if metadata else []
 if not layers:return schedule(env,oldrows,metadata,step)
 thresholds={(a,b):radii(env,a,b) for a,b in combinations(range(3),2)}
 for (a,b),v in list(thresholds.items()):thresholds[b,a]=v
 archive=[[[0.,*h,'W']] for h in env.homes];existing={};barrier=0.;details=[]
 for layer in layers:
  z=layer['z'];original=layer['layer_start_s'];end=original+layer['duration_s'];local={};geometry={};deposition_times={};lowers={}
  for r,rows in oldrows.items():
   rr=[[v[0]-original,*v[1:]] for v in rows if original-1e-7<=v[0]<=end+1e-7]
   if not rr or abs(rr[0][0])>1e-6 or abs(rr[-1][0]-layer['duration_s'])>1e-6:raise ValueError('Incomplete layer boundary')
   rr=trim_home_padding(rr,env.homes[r])
   if not rr:continue
   first=rr[0][0];rr=[[v[0]-first,*v[1:]] for v in rr];local[r]=rr;geometry[r]=reserved_parts(rr,env.bases[r,:2],env.homes[r],step)
   dd=[(a[0],b[0]) for a,b in zip(rr,rr[1:]) if a[4]=='D'];deposition_times[r]=(dd[0][0],dd[-1][1]);lowers[r]=max(0.,archive[r][-1][0],barrier-dd[0][0])
   for other in range(3):
    if r==other:continue
    p=np.asarray(env.homes[other,:2],np.float32).astype(float);park=([MultiPoint([p]).convex_hull],[MultiPoint([p,env.bases[other,:2]]).convex_hull])
    if len(pairs(geometry[r],park,thresholds[r,other],env.config.collision)):raise ValueError('Parked-home proof failed')
  lower=min(lowers.values())
  for other,g in list(existing.items()):
   ids=np.flatnonzero(g[2][:,1]>=lower-GUARD);existing[other]=([g[0][i] for i in ids],[g[1][i] for i in ids],g[2][ids])
  cache={};trials=[];best=None
  for r,new in geometry.items():
   oldblocks=[]
   for other,old in existing.items():
    if r==other:continue
    hit=pairs(new,old,thresholds[r,other],env.config.collision)
    if len(hit):oldblocks.append(forbidden(new[2],old[2],hit))
   cache[r]=oldblocks
  mm={(r,q):pairs(geometry[r],geometry[q],thresholds[r,q],env.config.collision) for r in local for q in local if r!=q}
  for order in permutations(sorted(local)):
   offsets={}
   for r in order:
    blocked=list(cache[r])
    for other,offset in offsets.items():
     hit=mm[r,other]
     if len(hit):blocked.append(forbidden(geometry[r][2],geometry[other][2]+offset,hit))
    offsets[r]=earliest_intervals(np.concatenate(blocked) if blocked else [],lowers[r])
   work_end=max(offsets[r]+deposition_times[r][1] for r in local);home_end=max(offsets[r]+local[r][-1][0] for r in local)
   score=(work_end,home_end,order);trials.append(dict(order=order,offsets=offsets,last_deposition_s=work_end,home_finish_s=home_end))
   if best is None or score<best[0]:best=(score,offsets)
  score,offsets=best;prior_barrier=barrier
  for r,rr in local.items():
   begin=offsets[r]
   if begin>archive[r][-1][0]+1e-9:archive[r][-1][4]='W';archive[r].append([begin,*env.homes[r],'W'])
   archive[r][-1][4]=rr[0][4];archive[r].extend([[v[0]+begin,*v[1:]] for v in rr[1:]])
   g=geometry[r];shifted=(g[0],g[1],g[2]+begin);old=existing.get(r);existing[r]=(old[0]+shifted[0],old[1]+shifted[1],np.concatenate((old[2],shifted[2]))) if old else shifted
  barrier=score[0];details.append(dict(z=z,offsets=offsets,previous_deposition_barrier_s=prior_barrier,last_deposition_s=barrier,home_finish_s=max(v[-1][0] for v in archive),priority_trials=trials))
  print(f'Independent layer {z}: work {barrier:.3f}s',flush=True)
 end=max(v[-1][0] for v in archive)
 for r in range(3):
  if archive[r][-1][0]<end:archive[r][-1][4]='W';archive[r].append([end,*env.homes[r],'W'])
 env.rows=archive;env.makespan=end
 return dict(kind='independent_layer_travel_overlap',layers=details,makespan_s=end,proof_step_mm=step,stationary_waits_reserved=True,within_robot_layer_timing_unchanged=True,selection_objective='earliest layer deposition completion then home completion')
