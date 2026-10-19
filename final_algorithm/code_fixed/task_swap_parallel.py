"""Fixed-assignment, fixed-direction task swaps with continuous reservations."""
import csv,json
from collections import defaultdict,deque
from itertools import combinations
import numpy as np
from shapely import MultiPoint,STRtree
from parallel_scheduler import parts,home_safe,radii
from timed_wait_parallel import conflict_pairs,schedule_robot
from flexible_parallel import earliest_intervals,GUARD
def explicit_model(env,routes,step):
 robots=sorted(routes)
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

def key(a,b):
 return tuple(sorted((tuple(np.asarray(a,np.float32)),tuple(np.asarray(b,np.float32)))))

def baseline_sequences(env,job,metadata):
 lookup=defaultdict(deque)
 for t in metadata['trace']:
  i=t['task'];lookup[t['robot_id']-1,key(env.tasks[i].start,env.tasks[i].end)].append(i)
 rows=defaultdict(list)
 with open(job/'trajectory.csv',newline='') as f:
  for v in csv.DictReader(f):rows[int(v['robot_id'])-1].append([float(v['time_s']),float(v['x_mm']),float(v['y_mm']),float(v['z_mm']),v['mode']])
 seq=defaultdict(lambda:defaultdict(list))
 for r,rr in rows.items():
  for a,b in zip(rr,rr[1:]):
   if a[4]!='D':continue
   i=lookup[r,key(a[1:4],b[1:4])].popleft()
   flip=tuple(np.asarray(a[1:4],np.float32))!=tuple(np.asarray(env.tasks[i].start,np.float32))
   seq[env.tasks[i].start[2]][r].append((i,flip))
 if any(lookup.values()):raise ValueError('Baseline task mapping incomplete')
 return seq,rows

def route(env,sequence,r):
 rows=[[0.,*env.homes[r],'W']]
 for i,flip in sequence:
  t=env.tasks[i];a,b=(t.end,t.start) if flip else (t.start,t.end)
  for p,mode,speed in ((a,'T',env.config.process.travel_speed_mm_s),(b,'D',env.config.process.deposition_speed_mm_s)):
   d=float(np.linalg.norm(np.asarray(p)-rows[-1][1:4]))
   if d>1e-9:rows[-1][4]=mode;rows.append([rows[-1][0]+d/speed,*p,'W'])
 d=float(np.linalg.norm(env.homes[r]-rows[-1][1:4]))
 if d>1e-9:rows[-1][4]='T';rows.append([rows[-1][0]+d/env.config.process.travel_speed_mm_s,*env.homes[r],'W'])
 return rows,[]

def evaluate(env,sequences,order,step):
 model=explicit_model(env,{r:route(env,s,r) for r,s in sequences.items()},step)
 assembled={};reserved={}
 for r in order:
  own=schedule_robot(model,r,False,reserved);reserved[r]=own;assembled[r]=own['rows']
 timed=dict(end=max(v[-1][0] for v in assembled.values()),rows=assembled)
 # Same priority and directions, compare a whole-route offset to waypoint waits.
 reserved={};assembled={}
 for r in order:
  times=model['geometry'][r][2];blocked=[]
  for other,data in reserved.items():
   arr=model['mm'][r,other]
   if len(arr):
    i,j=arr[:,0],arr[:,1];blocked.append(np.column_stack((data[j,0]-times[i,1]-GUARD,data[j,1]-times[i,0]+GUARD)))
  delay=earliest_intervals(np.concatenate(blocked) if blocked else [],0.)
  own=[[v[0]+delay,*v[1:]] for v in model['routes'][r][0]]
  if delay>0:own.insert(0,[0.,*env.homes[r],'W'])
  assembled[r]=own;reserved[r]=times+delay
 fixed=dict(end=max(v[-1][0] for v in assembled.values()),rows=assembled)
 return min((timed,fixed),key=lambda x:x['end'])

def swap_candidates(env,sequences,limit):
 candidates=[]
 for r,s in sequences.items():
  starts=[];ends=[]
  for i,f in s:
   t=env.tasks[i];a,b=(t.end,t.start) if f else (t.start,t.end);starts.append(np.asarray(a));ends.append(np.asarray(b))
  def edge(k,perm):
   a=env.homes[r] if k==0 else ends[perm.get(k-1,k-1)]
   b=env.homes[r] if k==len(s) else starts[perm.get(k,k)]
   return float(np.linalg.norm(a-b))
  for i in range(len(s)):
   for j in range(i+1,len(s)):
    touched={i,i+1,j,j+1};gain=sum(edge(k,{})-edge(k,{i:j,j:i}) for k in touched)
    candidates.append((gain,r,i,j))
 # Travel improvements first; deterministically inspect adjacent swaps too.
 candidates.sort(key=lambda v:(-v[0],v[1:]))
 chosen=candidates[:limit]
 adjacent=sorted((v for v in candidates if v[3]==v[2]+1),key=lambda v:(-v[0],v[1:]))
 for v in adjacent[:2]:
  if v not in chosen:chosen.append(v)
 return chosen

def schedule(env,baseline,metadata,step=10.,limit=4,rounds=2):
 sequences,oldrows=baseline_sequences(env,baseline,metadata)
 archive=[[[0.,*h,'W']] for h in env.homes];now=0.;details=[];trace=[]
 for layer in metadata['layers']:
  z=layer['z'];start=layer['layer_start_s'];end=start+layer['duration_s'];seq={r:list(s) for r,s in sequences[z].items()};order=layer['order']
  old={}
  for r,rr in oldrows.items():
   selected=[v.copy() for v in rr if start-1e-7<=v[0]<=end+1e-7]
   if not selected or abs(selected[0][0]-start)>1e-6 or abs(selected[-1][0]-end)>1e-6:raise ValueError('Missing layer barrier')
   old[r]=[[v[0]-start,*v[1:]] for v in selected]
  best=dict(end=layer['duration_s'],rows=old);accepted=[];trials=[]
  # The exact previous layer remains the incumbent, never reconstructed approximately.
  for iteration in range(rounds):
   winner=None
   for gain,r,i,j in swap_candidates(env,seq,limit):
    candidate={k:list(v) for k,v in seq.items()};candidate[r][i],candidate[r][j]=candidate[r][j],candidate[r][i]
    record=dict(round=iteration,robot_id=r+1,positions=[i,j],tasks=[seq[r][i][0],seq[r][j][0]],travel_gain_mm=gain)
    try:
     plan=evaluate(env,candidate,order,step);record['layer_s']=plan['end']
     if plan['end']<best['end']-1e-6 and (winner is None or plan['end']<winner[0]['end']):winner=(plan,candidate,record)
    except ValueError as e:record['rejected']=str(e)
    trials.append(record)
   if winner is None:break
   best,seq,record=winner;accepted.append(record)
  for r in range(3):
   own=[v.copy() for v in best['rows'].get(r,[[0.,*env.homes[r],'W'],[best['end'],*env.homes[r],'W']])]
   if own[-1][0]<best['end']-1e-9:own.append([best['end'],*env.homes[r],'W'])
   archive[r][-1][4]=own[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in own[1:]])
  for r,s in seq.items():trace.extend(dict(task=i,robot_id=r+1,layer_z=z) for i,f in s)
  details.append(dict(z=z,layer_start_s=now,duration_s=best['end'],baseline_s=layer['duration_s'],accepted_swaps=accepted,trials=trials,order=order))
  now+=best['end'];print(f'Swap layer {z}: {best["end"]:.3f}s; gain {layer["duration_s"]-best["end"]:.3f}s; swaps {len(accepted)}',flush=True)
 env.rows=archive;env.index=len(env.tasks);env.makespan=now;env.done=True
 return dict(trace=trace,layers=details,makespan_s=now,proof_step_mm=step,fixed_assignment=True,fixed_directions=True,fixed_priority=True,candidates_per_round=limit+2,max_rounds=rounds)
