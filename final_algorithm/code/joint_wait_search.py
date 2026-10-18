"""Wait-focused swaps and all robot priority permutations, independent of model ID."""
import csv
from itertools import permutations
from types import SimpleNamespace
import numpy as np
from task_swap_parallel import route,explicit_model
from timed_wait_parallel import schedule_robot
from flexible_parallel import earliest_intervals,GUARD
def evaluate_model(env,model,order):
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

def read_baseline(job):
 rows={r:[] for r in range(3)};raw=[];seq={};trace=[]
 with open(job/'trajectory.csv',newline='') as f:
  for v in csv.DictReader(f):rows[int(v['robot_id'])-1].append([float(v['time_s']),float(v['x_mm']),float(v['y_mm']),float(v['z_mm']),v['mode']])
 for r,rr in rows.items():
  for a,b in zip(rr,rr[1:]):
   if a[4]!='D':continue
   if abs(a[3]-b[3])>1e-6:raise ValueError('Non-planar deposition unsupported')
   i=len(raw);z=a[3];raw.append(dict(task_id=i,start_xyz_mm=a[1:4],end_xyz_mm=b[1:4]));seq.setdefault(z,{}).setdefault(r,[]).append((i,False));trace.append(dict(task=i,robot_id=r+1,layer_z=z))
 order=sorted(range(len(raw)),key=lambda i:(raw[i]['start_xyz_mm'][2],i));mapping={old:new for new,old in enumerate(order)}
 raw=[dict(raw[i],task_id=mapping[i]) for i in order]
 seq={z:{r:[(mapping[i],f) for i,f in ss] for r,ss in g.items()} for z,g in seq.items()}
 trace=[dict(v,task=mapping[v['task']]) for v in trace]
 return raw,seq,rows,trace

def wait_hotspots(rows,sequences):
 # Associate every positive wait with the next deposition task on that robot.
 result=[]
 for r,s in sequences.items():
  own=rows.get(r,[]);positions=[];wait=0.;p=0
  for a,b in zip(own,own[1:]):
   if a[4]=='W':wait+=b[0]-a[0]
   if a[4]=='D':
    if p<len(s):positions.append((wait,r,p))
    p+=1;wait=0.
  result.extend(v for v in positions if v[0]>1e-6)
 return sorted(result,key=lambda v:(-v[0],v[1],v[2]))

def targeted_candidates(env,sequences,rows,limit=3):
 chosen=[]
 # Two candidates prioritize actual waited tasks; move them before/after neighbors
 # or exchange with distant tasks. Keeps assignment and deposition direction.
 for wait,r,i in wait_hotspots(rows,sequences):
  n=len(sequences[r]);partners=[i-1,i+1,0,n-1,n//2]
  for j in partners:
   if 0<=j<n and j!=i:
    item=(r,min(i,j),max(i,j),'wait',wait)
    if not any(v[:3]==item[:3] for v in chosen):chosen.append(item);break
  if len(chosen)>=max(1,limit-1):break
 # One bounded travel candidate: adjacent exchange, linear cost scan.
 best=None
 for r,s in sequences.items():
  for i in range(len(s)-1):
   def distance(swapped):
    perm={i:i+1,i+1:i} if swapped else {};total=0.
    for k in (i,i+1,i+2):
     a=env.homes[r] if k==0 else env.tasks[s[perm.get(k-1,k-1)][0]].end
     b=env.homes[r] if k==len(s) else env.tasks[s[perm.get(k,k)][0]].start
     total+=float(np.linalg.norm(np.asarray(a)-b))
    return total
   gain=distance(False)-distance(True);item=(gain,r,i,i+1)
   if best is None or item>best:best=item
 if best:
  gain,r,i,j=best
  if not any(v[:3]==(r,i,j) for v in chosen):chosen.append((r,i,j,'travel',gain))
 return chosen[:limit]

def evaluate_all(env,sequences,step):
 # Geometry epsilon only needs to cover the current layer and all homes.
 local=SimpleNamespace(**env.__dict__);local.tasks=[env.tasks[i] for s in sequences.values() for i,f in s]
 model=explicit_model(local,{r:route(env,s,r) for r,s in sequences.items()},step)
 trials=[];best=None
 for order in permutations(sorted(sequences)):
  p=evaluate_model(env,model,order);trials.append(dict(order=order,duration_s=p['end']))
  if best is None or p['end']<best['end']-1e-9:best={**p,'order':order}
 best['priority_trials']=trials;return best

def schedule(env,sequences,oldrows,metadata=None,step=10.,limit=3):
 active={r for g in sequences.values() for r in g}
 if len(active)==1:return resident_search(env,sequences,step,limit)
 archive=[[[0.,*h,'W']] for h in env.homes];now=0.;details=[];trace=[]
 oldlayers={v['z']:v for v in metadata['layers']} if metadata else {}
 for z,seq in sorted(sequences.items()):
  seq={r:list(v) for r,v in seq.items()};old=oldlayers.get(z);incumbent=None
  # Exact home-barrier incumbent is available for existing layered schedules.
  if old:
   start=old['layer_start_s'];end=start+old['duration_s'];rr={r:[[v[0]-start,*v[1:]] for v in rows if start-1e-7<=v[0]<=end+1e-7] for r,rows in oldrows.items()}
   if all(rows and abs(rows[0][0])<1e-6 and abs(rows[-1][0]-old['duration_s'])<1e-6 for rows in rr.values()):incumbent=dict(end=old['duration_s'],rows=rr,order=old['order'])
  trials=[];errors=[];best=incumbent
  try:
   p=evaluate_all(env,seq,step);trials.append(dict(kind='priority_only',duration_s=p['end'],priority_trials=p['priority_trials']))
   if best is None or p['end']<best['end']-1e-6:best=p
  except ValueError as e:errors.append(str(e))
  # Measure waits in the actual current plan; if unavailable use original layer D/W.
  rr=best['rows'] if best else {r:[v for v in rows if abs(v[3]-z)<1e-6] for r,rows in oldrows.items()}
  candidates=targeted_candidates(env,seq,rr,limit);accepted=None;bestseq=seq
  for r,i,j,kind,score in candidates:
   changed={k:list(v) for k,v in seq.items()};changed[r][i],changed[r][j]=changed[r][j],changed[r][i]
   record=dict(kind=kind,robot_id=r+1,positions=[i,j],tasks=[seq[r][i][0],seq[r][j][0]],score=score)
   try:
    p=evaluate_all(env,changed,step);record.update(duration_s=p['end'],priority_trials=p['priority_trials'])
    if best is None or p['end']<best['end']-1e-6:best=p;bestseq=changed;accepted=record
   except ValueError as e:record['rejected']=str(e)
   trials.append(record)
  if best is None:raise ValueError(f'No proven home-barrier route at z={z}: {errors}')
  for r in range(3):
   own=[v.copy() for v in best['rows'].get(r,[[0.,*env.homes[r],'W'],[best['end'],*env.homes[r],'W']])]
   if own[-1][0]<best['end']-1e-9:own.append([best['end'],*env.homes[r],'W'])
   archive[r][-1][4]=own[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in own[1:]])
  for r,s in bestseq.items():trace.extend(dict(task=i,robot_id=r+1,layer_z=z) for i,f in s)
  details.append(dict(z=z,layer_start_s=now,duration_s=best['end'],order=best['order'],accepted_swap=accepted,trials=trials,rejections=errors))
  now+=best['end'];print(f'Joint layer {z}: {best["end"]:.3f}s; cumulative {now:.3f}s',flush=True)
 env.rows=archive;env.index=len(env.tasks);env.makespan=now;env.done=True
 return dict(trace=trace,layers=details,makespan_s=now,fixed_assignment=True,fixed_directions=True,priority_search='all permutations',max_swap_candidates_per_layer=limit,proof_step_mm=step)

def resident_search(env,sequences,step,limit):
 r=next(iter(next(iter(sequences.values()))));seq={z:list(g[r]) for z,g in sorted(sequences.items())};accepted=[]
 def flatten(ss):return [v for z,s in sorted(ss.items()) for v in s]
 current=route(env,flatten(seq),r)[0][-1][0]
 for z,ss in list(seq.items()):
  best=None
  for rr,i,j,kind,score in targeted_candidates(env,{r:ss},{},limit):
   changed={k:list(v) for k,v in seq.items()};changed[z][i],changed[z][j]=changed[z][j],changed[z][i]
   duration=route(env,flatten(changed),r)[0][-1][0]
   if duration<current-1e-6 and (best is None or duration<best[0]):best=(duration,changed,dict(z=z,robot_id=r+1,positions=[i,j],kind=kind))
  if best:current,seq,record=best;accepted.append(record)
 plan=evaluate_all(env,{r:flatten(seq)},step)
 end=plan['end'];archive=[]
 for robot in range(3):archive.append(plan['rows'][robot] if robot in plan['rows'] else [[0.,*env.homes[robot],'W'],[end,*env.homes[robot],'W']])
 env.rows=archive;env.index=len(env.tasks);env.makespan=end;env.done=True
 trace=[dict(task=i,robot_id=r+1,layer_z=z) for z,s in seq.items() for i,f in s]
 print(f'Single active robot resident search: {end:.3f}s, swaps {len(accepted)}',flush=True)
 return dict(trace=trace,layers=[],makespan_s=end,resident=True,accepted_swaps=accepted,fixed_assignment=True,fixed_directions=True,priority_search='single active robot',proof_step_mm=step,max_swap_candidates_per_layer=limit)
