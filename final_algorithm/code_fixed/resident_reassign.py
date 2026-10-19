"""Conservative resident parking and wait-aware, reach-checked reassignment."""
from itertools import combinations,permutations
from types import SimpleNamespace
import numpy as np
from shapely import MultiPoint,STRtree
from parallel_scheduler import parts,radii,eligible
from flexible_parallel import earliest_intervals,GUARD
from overlap_retimer import pairs,forbidden,schedule_independent
from joint_wait_search import evaluate_all

def full_thresholds(env):
 result={(a,b):radii(env,a,b) for a,b in combinations(range(3),2)}
 for (a,b),v in list(result.items()):result[b,a]=v
 return result

def point_shapes(point,base):
 p=np.asarray(point[:2],np.float32).astype(float)
 return MultiPoint([p]).convex_hull,MultiPoint([p,base]).convex_hull

def parked_proof(point,base,trees,thresholds,collision):
 shapes=point_shapes(point,base)
 for kind,enabled in ((0,collision.check_tcp_radius),(1,collision.check_arm_envelope)):
  if enabled and len(trees[kind].query_nearest(shapes[kind],max_distance=thresholds[kind],all_matches=False)):return False
 return True

def explicit_route(env,sequence,r,start,finish):
 rows=[[0.,*start,'W']]
 def move(p,mode,speed):
  d=float(np.linalg.norm(np.asarray(p)-rows[-1][1:4]))
  if d>1e-9:rows[-1][4]=mode;rows.append([rows[-1][0]+d/speed,*p,'W'])
 for i,f in sequence:
  t=env.tasks[i];a,b=(t.end,t.start) if f else (t.start,t.end)
  move(a,'T',env.config.process.travel_speed_mm_s);move(b,'D',env.config.process.deposition_speed_mm_s)
 move(finish,'T',env.config.process.travel_speed_mm_s)
 return rows

def resident_layer(env,sequences,current,endpoints,thresholds,step):
 routes={r:explicit_route(env,sequences.get(r,[]),r,current[r],endpoints[r]) for r in range(3)}
 geometry={r:parts(rr,env.bases[r,:2],step) for r,rr in routes.items() if len(rr)>1}
 for r,g in geometry.items():
  for other in range(3):
   if r==other:continue
   for point in (current[other],endpoints[other]):
    park=point_shapes(point,env.bases[other,:2]);fixed=([park[0]],[park[1]])
    if len(pairs(g,fixed,thresholds[r,other],env.config.collision)):raise ValueError('Start/end work parking cannot be proven safe')
 mm={(r,q):pairs(geometry[r],geometry[q],thresholds[r,q],env.config.collision) for r in geometry for q in geometry if r!=q};best=None
 for order in permutations(sorted(geometry)):
  starts={}
  for r in order:
   blocked=[]
   for q,offset in starts.items():
    hit=mm[r,q]
    if len(hit):blocked.append(forbidden(geometry[r][2],geometry[q][2]+offset,hit))
   starts[r]=earliest_intervals(np.concatenate(blocked) if blocked else [],0.)
  rows={};ends=[]
  for r,rr in routes.items():
   off=starts.get(r,0.);own=[[v[0]+off,*v[1:]] for v in rr]
   if off>0:own.insert(0,[0.,*current[r],'W'])
   rows[r]=own;ends.append(own[-1][0])
  score=(max(ends),sum(ends),order)
  if best is None or score<best[0]:best=(score,rows,order)
 return dict(end=best[0][0],rows=best[1],order=best[2],secondary=best[0][1])

def schedule_resident(env,sequences,oldrows,step=10.):
 thresholds=full_thresholds(env);trees={}
 for r,rr in oldrows.items():
  g=parts(rr,env.bases[r,:2],1e30);p=point_shapes(env.homes[r],env.bases[r,:2]);trees[r]=(STRtree(g[0]+[p[0]]),STRtree(g[1]+[p[1]]))
 current=env.homes.copy();archive=[[[0.,*h,'W']] for h in env.homes];now=0.;details=[];env.progress=details;zs=sorted(sequences)
 for layer,z in enumerate(zs):
  seq=sequences[z];possible=[];certificates=[]
  if layer<len(zs)-1:
   for r,ss in seq.items():
    i,f=ss[-1];point=np.asarray(env.tasks[i].start if f else env.tasks[i].end,float)
    safe=all(parked_proof(point,env.bases[r,:2],trees[q],thresholds[r,q],env.config.collision) for q in range(3) if q!=r)
    certificates.append(dict(robot_id=r+1,certified=safe))
    if safe:possible.append((float(np.linalg.norm(point-env.homes[r])),r,point))
  # Bound branching to home plus one largest-detour certified resident endpoint.
  choices=[('home',env.homes.copy())]
  if possible:
   _,r,p=max(possible,key=lambda v:(v[0],-v[1]));ends=env.homes.copy();ends[r]=p;choices.append(('resident',ends))
  plans=[];errors=[]
  for label,ends in choices:
   try:
    plan=resident_layer(env,seq,current,ends,thresholds,step);plan.update(label=label,endpoints=ends);plans.append(plan)
   except ValueError as e:errors.append(dict(kind=label,error=str(e)))
  if not plans:
   env.failed_layer=dict(z=z,parking_certificates=certificates,errors=errors)
   raise ValueError(f'No certified resident/home plan at layer {z}: {errors}')
  best=min(plans,key=lambda p:(p['end'],p['secondary'],p['label']));end=best['end']
  for r in range(3):
   own=best['rows'][r]
   if own[-1][0]<end:own[-1][4]='W';own.append([end,*best['endpoints'][r],'W'])
   archive[r][-1][4]=own[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in own[1:]])
  details.append(dict(z=z,layer_start_s=now,duration_s=end,kind=best['label'],order=best['order'],parking_certificates=certificates,errors=errors))
  current=best['endpoints'];now+=end;print(f'Resident layer {z}: {end:.3f}s; {best["label"]}',flush=True)
 env.rows=archive;env.makespan=now
 return dict(kind='certified_resident',layers=details,makespan_s=now,proof_step_mm=step,fixed_assignment=True)

def insertion(env,sequence,item,r):
 t=env.tasks[item[0]];a,b=(t.end,t.start) if item[1] else (t.start,t.end);best=None
 for k in range(len(sequence)+1):
  if k:tt=env.tasks[sequence[k-1][0]];prev=tt.start if sequence[k-1][1] else tt.end
  else:prev=env.homes[r]
  if k<len(sequence):tt=env.tasks[sequence[k][0]];nxt=tt.end if sequence[k][1] else tt.start
  else:nxt=env.homes[r]
  delta=float(np.linalg.norm(np.asarray(a)-prev)+np.linalg.norm(np.asarray(nxt)-b)-np.linalg.norm(np.asarray(nxt)-prev))
  candidate=(delta,k)
  if best is None or candidate<best:best=candidate
 result=list(sequence);result.insert(best[1],item);return result,best[0]

def wait_scores(oldrows,sequences):
 result={};finishes={}
 for r,rr in oldrows.items():
  ids=[(z,i) for z,g in sorted(sequences.items()) for i,f in g.get(r,[])];index=0;wait=0.
  for a,b in zip(rr,rr[1:]):
   if a[4]=='W':wait+=b[0]-a[0]
   if a[4]=='D':
    z,i=ids[index];result[i]=wait;finishes[z,r]=b[0];wait=0.;index+=1
 return result,finishes

def assignment_candidates(env,seq,wait,finish,z):
 source=max(seq,key=lambda r:(finish.get((z,r),0.),r));ranked=[]
 for i,f in seq[source]:
  t=env.tasks[i];duration=float(np.linalg.norm(np.asarray(t.end)-t.start))/env.config.process.deposition_speed_mm_s
  ranked.append((wait.get(i,0.)+duration,i,f))
 ranked.sort(key=lambda v:(-v[0],v[1]));moves=[]
 # Shortlist eight high-delay/load tasks, then rank destination insertion cost.
 for urgency,i,f in ranked[:8]:
  for target in range(3):
   if target==source or not eligible(env,[i],target):continue
   ins,cost=insertion(env,seq.get(target,[]),(i,f),target)
   delay=max(0.,finish.get((z,target),0.)-finish.get((z,source),0.))
   score=urgency-cost/env.config.process.travel_speed_mm_s-delay
   moves.append((score,source,target,i,f,ins))
 if not moves:return []
 moves.sort(key=lambda v:(-v[0],v[1:5]));_,source,target,i,f,ins=moves[0]
 moved={r:list(v) for r,v in seq.items()};moved[source].remove((i,f));moved[target]=ins;moved={r:v for r,v in moved.items() if v}
 result=[(moved,dict(kind='transfer',task=i,from_robot=source+1,to_robot=target+1))]
 # A reverse transfer tests whether two-way task exchange reduces arm conflicts.
 back=[]
 for j,g in seq.get(target,[]):
  if not eligible(env,[j],source):continue
  rest=[v for v in seq[source] if v!=(i,f)];ss,cost=insertion(env,rest,(j,g),source)
  back.append((cost,j,g,ss))
 if back:
  _,j,g,ss=min(back,key=lambda v:v[:3]);swapped={r:list(v) for r,v in moved.items()};swapped[source]=ss;swapped[target]=[v for v in ins if v!=(j,g)];swapped={r:v for r,v in swapped.items() if v};result.append((swapped,dict(kind='exchange',tasks=[i,j],robots=[source+1,target+1])))
 return result

def schedule_reassign(env,sequences,oldrows,source_rows,metadata,step=10.):
 if not metadata.get('layers'):
  env.rows=[[v.copy() for v in oldrows[r]] for r in range(3)];env.makespan=max(v[-1][0] for v in env.rows)
  return dict(kind='preserved_no_home_layer_metadata',layers=[],makespan_s=env.makespan)
 wait,finish=wait_scores(oldrows,sequences);archive=[[[0.,*h,'W']] for h in env.homes];now=0.;details=[];trace=[];env.progress=details
 for layer in metadata['layers']:
  z=layer['z'];seq={r:list(v) for r,v in sequences[z].items()};start=layer['layer_start_s'];end=start+layer['duration_s']
  rr={r:[[v[0]-start,*v[1:]] for v in rows if start-1e-7<=v[0]<=end+1e-7] for r,rows in source_rows.items()}
  if not all(v and abs(v[0][0])<1e-6 and abs(v[-1][0]-layer['duration_s'])<1e-6 for v in rr.values()):raise ValueError('Missing preserved source layer boundary')
  best=dict(end=layer['duration_s'],rows=rr,order=layer['order']);bestseq=seq;accepted=None;trials=[]
  for candidate,change in assignment_candidates(env,seq,wait,finish,z):
   record=dict(change)
   try:
    p=evaluate_all(env,candidate,step);record.update(duration_s=p['end'],priority_trials=p['priority_trials'])
    if p['end']<best['end']-1e-6:best=p;bestseq=candidate;accepted=change
   except ValueError as e:record['rejected']=str(e)
   trials.append(record)
  for r in range(3):
   own=[v.copy() for v in best['rows'].get(r,[[0.,*env.homes[r],'W'],[best['end'],*env.homes[r],'W']])]
   if own[-1][0]<best['end']:own[-1][4]='W';own.append([best['end'],*env.homes[r],'W'])
   archive[r][-1][4]=own[0][4];archive[r].extend([[v[0]+now,*v[1:]] for v in own[1:]])
  details.append(dict(z=z,layer_start_s=now,duration_s=best['end'],order=best['order'],accepted_reassignment=accepted,trials=trials))
  for r,ss in bestseq.items():trace.extend(dict(task=i,robot_id=r+1,layer_z=z) for i,f in ss)
  now+=best['end'];print(f'Reassign layer {z}: {best["end"]:.3f}s; changed {accepted is not None}',flush=True)
 # Apply the already tested cross-layer travel overlap to the new assignment plan.
 diagnostics=dict(trace=trace,layers=details,makespan_s=now)
 env.rows=archive;env.makespan=now;home_rows={r:[v.copy() for v in rows] for r,rows in enumerate(archive)}
 overlap=schedule_independent(env,home_rows,diagnostics,step)
 return dict(kind='wait_aware_reassignment_and_overlap',layers=details,overlap=overlap,trace=trace,makespan_s=env.makespan,proof_step_mm=step,max_assignment_candidates_per_layer=2)
