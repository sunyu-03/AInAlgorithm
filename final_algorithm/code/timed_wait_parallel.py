"""Time-window-safe waits, including reserved waits of other robots.

Fixed start-offset schedules are retained as incumbents. A bounded retry
delays entry from home when waiting at a work vertex would be unsafe.
"""
from itertools import combinations,product,permutations
import numpy as np
from shapely import MultiPoint,STRtree
from algorithm.parallel_layers import nearest_route
from parallel_scheduler import parts,home_safe,radii
from refine_parallel import reverse_rows
from flexible_parallel import nearest_groups,rebalance,earliest_intervals,GUARD

def conflict_pairs(env,a,b,left,right):
 tcp,arm=radii(env,a,b);c=env.config.collision;found=[]
 for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
  if enabled:
   p=STRtree(right[kind]).query(left[kind],predicate='dwithin',distance=radius)
   if p.shape[1]:found.append(p.T)
 return np.unique(np.concatenate(found),axis=0) if found else np.empty((0,2),int)

def static_model(env,groups,step):
 routes={r:nearest_route(env,g,r) for r,g in groups.items()};robots=sorted(routes)
 geometry={r:parts(v[0],env.bases[r,:2],step) for r,v in routes.items()};vertices={};leg_parts={};piece_leg={}
 for r,g in geometry.items():
  if not home_safe(env,r,g):raise ValueError('Parked-home separation cannot be proven')
  rows=routes[r][0];xy=np.asarray([v[1:3] for v in rows],np.float32).astype(float)
  vertices[r]=([MultiPoint([p]).convex_hull for p in xy],[MultiPoint([p,env.bases[r,:2]]).convex_hull for p in xy])
  mapping=np.searchsorted(np.array([v[0] for v in rows]),g[2][:,0]+1e-7,side='right')-1
  piece_leg[r]=mapping;leg_parts[r]={k:np.flatnonzero(mapping==k) for k in range(len(rows)-1)}
 mm={};vm={};vv={};leg_mm={};leg_mv={};vertex_vm={};vertex_vv={}
 for a,b in combinations(robots,2):
  arr=conflict_pairs(env,a,b,geometry[a],geometry[b]);mm[a,b]=arr;mm[b,a]=arr[:,::-1]
  arr=conflict_pairs(env,a,b,vertices[a],vertices[b]);vv[a,b]=arr;vv[b,a]=arr[:,::-1]
 for a in robots:
  for b in robots:
   if a==b:continue
   vm[a,b]=conflict_pairs(env,a,b,vertices[a],geometry[b])
 for a in robots:
  for b in robots:
   if a==b:continue
   for k in leg_parts[a]:
    arr=mm[a,b];leg_mm[a,b,k]=arr[piece_leg[a][arr[:,0]]==k] if len(arr) else arr
    arr=vm[b,a][:,::-1];leg_mv[a,b,k]=arr[piece_leg[a][arr[:,0]]==k] if len(arr) else arr
   for v in range(len(routes[a][0])):
    vertex_vm[a,b,v]=vm[a,b][vm[a,b][:,0]==v,1]
    vertex_vv[a,b,v]=vv[a,b][vv[a,b][:,0]==v,1]
 return dict(routes=routes,robots=robots,geometry=geometry,leg_parts=leg_parts,mm=mm,leg_mm=leg_mm,leg_mv=leg_mv,vertex_vm=vertex_vm,vertex_vv=vertex_vv)

def wait_conflict_end(model,r,vertex,arrival,departure,reserved):
 end=None
 for other,data in reserved.items():
  ids=model['vertex_vm'][r,other,vertex];intervals=data['times'][ids] if len(ids) else np.empty((0,2))
  if len(intervals):
   hit=intervals[(intervals[:,0]<=departure+GUARD)&(intervals[:,1]>=arrival-GUARD)]
   if len(hit):end=max(end if end is not None else -np.inf,float(hit[:,1].max())+GUARD)
  unsafe=model['vertex_vv'][r,other,vertex]
  for v,a,b in data['waits']:
   if v in unsafe and a<=departure+GUARD and b>=arrival-GUARD:end=max(end if end is not None else -np.inf,b+GUARD)
 return end

def leg_start(model,r,leg,idx,times,left,reserved,ready):
 blocked=[]
 for other,data in reserved.items():
  hit=model['leg_mm'][r,other,leg]
  if len(hit):
   i,j=hit[:,0],hit[:,1];old=data['times']
   blocked.append(np.column_stack((old[j,0]-(times[i,1]-left)-GUARD,old[j,1]-(times[i,0]-left)+GUARD)))
  hit=model['leg_mv'][r,other,leg]
  for v,a,b in data['waits']:
   i=hit[hit[:,1]==v,0]
   if len(i):blocked.append(np.column_stack((a-(times[i,1]-left)-GUARD,b-(times[i,0]-left)+GUARD)))
 return earliest_intervals(np.concatenate(blocked) if blocked else [],ready)

def schedule_robot(model,r,flip,reserved,retries=8):
 original=model['routes'][r][0];duration=original[-1][0];source=reverse_rows(original) if flip else original
 times=duration-model['geometry'][r][2][:,::-1] if flip else model['geometry'][r][2]
 release=0.;restarts=0
 for attempt in range(retries+1):
  if attempt==retries:
   release=max((float(d['times'][:,1].max()) for d in reserved.values()),default=0.)+2*GUARD
  own=[[0.,*source[0][1:4],'W']];ready=release;actual=np.zeros_like(times);waits=[];failed=False
  if release>0:own.append([release,*source[0][1:4],'W'])
  for k,(a,b) in enumerate(zip(source,source[1:])):
   leg=len(original)-2-k if flip else k;vertex=len(original)-1-k if flip else k
   idx=model['leg_parts'][r][leg]
   start=leg_start(model,r,leg,idx,times,a[0],reserved,ready)
   if start>ready+1e-9:
    end=wait_conflict_end(model,r,vertex,ready,start,reserved)
    if end is not None:
     release+=max(end-ready+2*GUARD,GUARD);restarts+=1;failed=True;break
    own[-1][4]='W';own.append([start,*a[1:4],'W']);waits.append((vertex,ready,start))
   own[-1][4]=a[4];ready=start+b[0]-a[0];own.append([ready,*b[1:]])
   actual[idx]=times[idx]+start-a[0]
  if not failed:return dict(rows=own,times=actual,waits=waits,restarts=restarts)
 raise ValueError('No time-safe route, including serial fallback')

def evaluate(env,groups,step,reverse_search=True,top=6):
 model=static_model(env,groups,step);robots=model['robots'];routes=model['routes'];durations={r:routes[r][0][-1][0] for r in robots};fixed=[]
 for flags in product((False,True) if reverse_search else (False,),repeat=len(robots)):
  flip=dict(zip(robots,flags));times={r:durations[r]-model['geometry'][r][2][:,::-1] if flip[r] else model['geometry'][r][2] for r in robots}
  for order in permutations(robots):
   reserved={};assembled={}
   for r in order:
    blocked=[]
    for other,data in reserved.items():
     arr=model['mm'][r,other]
     if len(arr):
      i,j=arr[:,0],arr[:,1];old=data['times'];blocked.append(np.column_stack((old[j,0]-times[r][i,1]-GUARD,old[j,1]-times[r][i,0]+GUARD)))
    start=earliest_intervals(np.concatenate(blocked) if blocked else [],0.)
    source=reverse_rows(routes[r][0]) if flip[r] else routes[r][0];rows=[[v[0]+start,*v[1:]] for v in source]
    if start>0:rows.insert(0,[0.,*env.homes[r],'W'])
    assembled[r]=rows;reserved[r]=dict(times=times[r]+start,waits=[])
   end=max(v[-1][0] for v in assembled.values());fixed.append(dict(end=end,rows=assembled,flags=flags,order=order,mid_wait=False,wait_events=0,restarts=0,groups=groups))
 fixed.sort(key=lambda v:(v['end'],v['flags'],v['order']));candidates=fixed[:1]
 for base in fixed[:top]:
  flags=base['flags'];flip=dict(zip(robots,flags));reserved={};assembled={};restarts=0;waits=0
  for r in base['order']:
   own=schedule_robot(model,r,flip[r],reserved);reserved[r]=own;assembled[r]=own['rows'];restarts+=own['restarts'];waits+=len(own['waits'])
  candidates.append(dict(end=max(v[-1][0] for v in assembled.values()),rows=assembled,flags=flags,order=base['order'],mid_wait=True,wait_events=waits,restarts=restarts,groups=groups))
 best=min(candidates,key=lambda v:(v['end'],v['mid_wait'],v['flags'],v['order']));best['trials']=len(fixed)+len(candidates)-1
 return best

def schedule(env,step=10.,reverse_search=True):
 layers={};archive=[[[0.,*h,'W']] for h in env.homes];now=0.;trace=[];details=[]
 for i,t in enumerate(env.tasks):layers.setdefault(t.start[2],[]).append(i)
 for z,indices in sorted(layers.items()):
  groups=nearest_groups(env,indices);balanced,moves=rebalance(env,groups);plans=[];errors=[]
  for label,g,changes in [('nearest',groups,[]),('balanced',balanced,moves)]:
   if label=='balanced' and not changes:continue
   try:
    plan=evaluate(env,g,step,reverse_search);plan.update(assignment=label,moves=changes);plans.append(plan)
   except ValueError as e:errors.append(dict(assignment=label,error=str(e)))
  if not plans:raise ValueError(f'No safe plan at layer {z}: {errors}')
  plan=min(plans,key=lambda v:(v['end'],v['assignment']));end=plan['end']
  for r in range(3):
   own=[v.copy() for v in plan['rows'][r]] if r in plan['rows'] else [[0.,*env.homes[r],'W'],[end,*env.homes[r],'W']]
   if own[-1][0]<end:own.append([end,*env.homes[r],'W'])
   archive[r][-1][4]=own[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in own[1:]])
  for r,g in plan['groups'].items():trace.extend(dict(task=i,robot_id=r+1,layer_z=z) for i in g)
  details.append(dict(z=z,layer_start_s=now,duration_s=end,assignment=plan['assignment'],moves=plan['moves'],mid_wait=plan['mid_wait'],wait_events=plan['wait_events'],restarts=plan['restarts'],order=plan['order'],flags=plan['flags'],trials=sum(v['trials'] for v in plans),rejected_assignments=errors))
  now+=end;print(f'Timed wait layer {z}: layer_s={end:.3f}, total_s={now:.3f}, assignment={plan["assignment"]}, mid_wait={plan["mid_wait"]}',flush=True)
 env.rows=archive;env.index=len(env.tasks);env.makespan=now;env.done=True
 return dict(trace=trace,layers=details,makespan_s=now,proof_step_mm=step,reverse_search=reverse_search)
