"""Bounded reassignment and safe waypoint waiting with continuous proofs.

Waiting vertices must be separated from every other robot's entire path,
not just its sampled/current position. Deposition beads remain unchanged.
"""
from itertools import combinations,product,permutations
import numpy as np
from shapely import MultiPoint,STRtree
from algorithm.parallel_layers import nearest_route
from parallel_scheduler import parts,home_safe,radii,eligible
from refine_parallel import reverse_rows

GUARD=1e-6
def earliest_intervals(intervals,lower=0.):
 if len(intervals)==0:return max(0.,lower)
 a=np.asarray(intervals,float).reshape(-1,2);order=np.lexsort((a[:,1],a[:,0]));a=a[order]
 rights=np.maximum.accumulate(a[:,1]);start=max(0.,lower)
 while True:
  j=np.searchsorted(a[:,0],start,side='right')-1
  if j<0 or rights[j]<start:return float(start)
  start=float(rights[j]+GUARD)

def nearest_groups(env,indices):
 groups={}
 for i in indices:
  valid=[r for r in range(3) if eligible(env,[i],r)]
  if not valid:raise ValueError(f'Unreachable task {i}')
  mid=(np.asarray(env.tasks[i].start)+env.tasks[i].end)/2
  r=min(valid,key=lambda r:(np.linalg.norm(mid[:2]-env.bases[r,:2]),r));groups.setdefault(r,[]).append(i)
 return groups

def rebalance(env,groups,moves=4):
 result={r:list(g) for r,g in groups.items()};loads=np.zeros(3)
 duration={i:float(np.linalg.norm(np.asarray(env.tasks[i].end)-env.tasks[i].start)/env.config.process.deposition_speed_mm_s) for g in groups.values() for i in g}
 for r,g in result.items():loads[r]=sum(duration[i] for i in g)
 trace=[]
 for _ in range(moves):
  current=(float(loads.max()),float(loads@loads));choices=[]
  for r,g in sorted(result.items()):
   for i in sorted(g):
    for other in range(3):
     if other==r or not eligible(env,[i],other):continue
     trial=loads.copy();trial[r]-=duration[i];trial[other]+=duration[i]
     score=(float(trial.max()),float(trial@trial))
     if score<current:choices.append((score,r,i,other,trial))
  if not choices:break
  _,r,i,other,loads=min(choices,key=lambda c:c[:4]);result[r].remove(i);result.setdefault(other,[]).append(i)
  trace.append(dict(task=i,from_robot=r+1,to_robot=other+1))
 return {r:g for r,g in result.items() if g},trace

def waypoint_safety(env,r,rows,geometry):
 points=np.asarray([v[1:3] for v in rows],np.float32).astype(float)
 candidates=([MultiPoint([p]).convex_hull for p in points],[MultiPoint([p,env.bases[r,:2]]).convex_hull for p in points])
 safe=np.ones(len(rows),bool);c=env.config.collision
 for other,old in geometry.items():
  if other==r:continue
  tcp,arm=radii(env,r,other)
  for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
   if enabled:
    found=STRtree(old[kind]).query(candidates[kind],predicate='dwithin',distance=radius)
    safe[found[0]]=False
 return safe

def evaluate_groups(env,groups,step,reverse_search=True):
 routes={r:nearest_route(env,g,r) for r,g in groups.items()};robots=sorted(routes)
 durations={r:v[0][-1][0] for r,v in routes.items()};geometry={r:parts(v[0],env.bases[r,:2],step) for r,v in routes.items()}
 for r,g in geometry.items():
  if not home_safe(env,r,g):raise ValueError('Parked-home proof failed')
 safety={r:waypoint_safety(env,r,routes[r][0],geometry) for r in robots}
 if not all(safety[r][0] and safety[r][-1] for r in robots):raise ValueError('Home waypoint not proven safe')
 pairs={};c=env.config.collision
 for a,b in combinations(robots,2):
  tcp,arm=radii(env,a,b);found_pairs=set()
  for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
   if enabled:
    found=STRtree(geometry[b][kind]).query(geometry[a][kind],predicate='dwithin',distance=radius)
    found_pairs.update(zip(found[0].tolist(),found[1].tolist()))
  arr=np.asarray(sorted(found_pairs),int).reshape(-1,2);pairs[a,b]=arr;pairs[b,a]=arr[:,::-1]
 best=None;trials=0
 for flags in product((False,True) if reverse_search else (False,),repeat=len(robots)):
  reversal=dict(zip(robots,flags));rows={r:reverse_rows(routes[r][0]) if reversal[r] else routes[r][0] for r in robots}
  times={r:durations[r]-geometry[r][2][:,::-1] if reversal[r] else geometry[r][2] for r in robots}
  chunks={}
  for r in robots:
   safe=safety[r][::-1] if reversal[r] else safety[r];cuts=np.flatnonzero(safe).tolist();own=rows[r];chunks[r]=[]
   for lo,hi in zip(cuts,cuts[1:]):
    left,right=own[lo][0],own[hi][0]
    idx=np.flatnonzero((times[r][:,0]>=left-1e-7)&(times[r][:,1]<=right+1e-7))
    chunks[r].append((lo,hi,left,right,idx))
  for order in permutations(robots):
   for flexible in (False,True):
    reserved={};assembled={};wait_count=0
    for r in order:
     source=rows[r];selected_chunks=chunks[r] if flexible else [(0,len(source)-1,0.,durations[r],np.arange(len(times[r])))]
     own=[[0.,*source[0][1:4],'W']];actual=np.zeros_like(times[r]);ready=0.
     for lo,hi,left,right,idx in selected_chunks:
      blocked=[]
      for other,old_times in reserved.items():
       arr=pairs[r,other]
       if not len(arr):continue
       hit=arr[np.isin(arr[:,0],idx)]
       if len(hit):
        ii,jj=hit[:,0],hit[:,1]
        blocked.append(np.column_stack((old_times[jj,0]-(times[r][ii,1]-left)-GUARD,old_times[jj,1]-(times[r][ii,0]-left)+GUARD)))
      start=earliest_intervals(np.concatenate(blocked) if blocked else [],ready)
      if start>ready+1e-9:own[-1][4]='W';own.append([start,*source[lo][1:4],'W']);wait_count+=1
      own[-1][4]=source[lo][4]
      own.extend([[start+v[0]-left,*v[1:]] for v in source[lo+1:hi+1]])
      actual[idx]=times[r][idx]+start-left;ready=start+right-left
     reserved[r]=actual;assembled[r]=own
    end=max(v[-1][0] for v in assembled.values());trials+=1
    key=(end,flags,tuple(order),flexible)
    if best is None or key<best['key']:
     best=dict(key=key,end=end,rows=assembled,reversal=reversal,order=order,mid_wait=flexible,wait_events=wait_count,safe_waypoints={r:int(safety[r].sum()) for r in robots},groups=groups)
 best['trials']=trials
 return best

def schedule(env,step=10.,reverse_search=True):
 layers={};archive=[[[0.,*h,'W']] for h in env.homes];now=0.;trace=[];details=[]
 for i,t in enumerate(env.tasks):layers.setdefault(t.start[2],[]).append(i)
 for z,indices in sorted(layers.items()):
  base=nearest_groups(env,indices);balanced,moves=rebalance(env,base)
  choices=[];errors=[]
  for label,groups,changes in [('nearest',base,[]),('balanced',balanced,moves)]:
   if label=='balanced' and not changes:continue
   try:
    plan=evaluate_groups(env,groups,step,reverse_search);plan.update(assignment=label,moves=changes);choices.append(plan)
   except ValueError as e:errors.append(dict(assignment=label,error=str(e)))
  if not choices:raise ValueError(f'No safe layer plan at z={z}: {errors}')
  plan=min(choices,key=lambda v:(v['end'],v['assignment']));end=plan['end']
  for r in range(3):
   local=[v.copy() for v in plan['rows'][r]] if r in plan['rows'] else [[0.,*env.homes[r],'W'],[end,*env.homes[r],'W']]
   if local[-1][0]<end:local.append([end,*env.homes[r],'W'])
   archive[r][-1][4]=local[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in local[1:]])
  for r,g in plan['groups'].items():trace.extend(dict(task=i,robot_id=r+1,layer_z=z) for i in g)
  details.append(dict(z=z,layer_start_s=now,duration_s=end,assignment=plan['assignment'],moves=plan['moves'],mid_wait=plan['mid_wait'],wait_events=plan['wait_events'],safe_waypoints=plan['safe_waypoints'],order=plan['order'],reversal=plan['reversal'],trials=sum(v['trials'] for v in choices),rejected_assignments=errors))
  now+=end;print(f'Flexible layer {z}: layer_s={end:.3f}, total_s={now:.3f}, assignment={plan["assignment"]}, mid_wait={plan["mid_wait"]}',flush=True)
 env.rows=archive;env.index=len(env.tasks);env.makespan=now;env.done=True
 return dict(trace=trace,layers=details,makespan_s=now,proof_step_mm=step,reverse_search=reverse_search)
