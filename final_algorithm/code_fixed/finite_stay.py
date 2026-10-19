"""Replace inter-layer home detours using timed certificates, then remove global idle.

Only globally stationary time intervals are removed. No simultaneous motion
is reordered or sped up. Deposition CSV legs are never subdivided.
"""
import numpy as np
from shapely import STRtree
from parallel_scheduler import parts
from resident_reassign import full_thresholds,point_shapes
from overlap_retimer import forbidden
GUARD=1e-6

def merge(intervals):
 result=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if result and a<=result[-1][1]+1e-9:result[-1][1]=max(result[-1][1],b)
  else:result.append([float(a),float(b)])
 return result

def all_parts(rows,base,step):
 tcp,arm,times=parts(rows,base,step);times=list(times)
 for a,b in zip(rows,rows[1:]):
  if a[4]=='W' and b[0]>a[0]:
   p=point_shapes(a[1:4],base);tcp.append(p[0]);arm.append(p[1]);times.append((a[0],b[0]))
 return tcp,arm,np.asarray(times,float).reshape(-1,2)

def hits(new,trees,threshold,collision):
 found=[]
 for k,enabled in ((0,collision.check_tcp_radius),(1,collision.check_arm_envelope)):
  if enabled and len(new[k]):
   a=trees[k].query(new[k],predicate='dwithin',distance=threshold[k])
   if a.shape[1]:found.append(a.T)
 return np.unique(np.concatenate(found),axis=0) if found else np.empty((0,2),int)

def window(rows,start,end):
 times=np.asarray([v[0] for v in rows]);i=max(0,int(np.searchsorted(times,start,side='right'))-1);j=min(len(rows),int(np.searchsorted(times,end,side='left'))+1)
 return rows[i:j]

def allowed_intervals(blocked,lower,upper):
 if upper<lower-1e-9:return []
 result=[];pos=lower
 for a,b in merge(blocked):
  if b<pos:continue
  if a>upper:break
  if a>pos+GUARD:result.append((pos,min(upper,a-GUARD)))
  pos=max(pos,b+GUARD)
  if pos>upper:break
 if pos<=upper:result.append((pos,upper))
 return result

def best_departure(intervals,duration,motions):
 # The overlap objective is piecewise linear; these are its breakpoints.
 moves=merge(motions);edges=[v for aa in moves for v in aa];points=edges+[v-duration for v in edges]
 starts=np.asarray([a for a,b in moves]);ends=np.asarray([b for a,b in moves]);cum=np.r_[0.,np.cumsum(ends-starts)]
 def integral(t):
  if not len(starts):return 0.
  j=np.searchsorted(starts,t,side='right')-1
  return 0. if j<0 else float(cum[j]+min(max(0.,t-starts[j]),ends[j]-starts[j]))
 choices=[]
 for lo,hi in intervals:
  for t in [lo,hi]+[p for p in points if lo<=p<=hi]:choices.append((-(integral(t+duration)-integral(t)),t))
 return min(choices)[1] if choices else None

def gap_rows(a,b,p,q,depart,duration,end_mode):
 rr=[[a,*p,'W']]
 if duration>1e-10:
  if depart>a+1e-9:rr.append([depart,*p,'T'])
  else:rr[-1][4]='T'
  rr.append([depart+duration,*q,'W'])
 if b>rr[-1][0]+1e-9:rr.append([b,*q,end_mode])
 else:rr[-1][4]=end_mode
 return rr

def propose(env,r,start,end,rows,thresholds,step):
 a,p=start[0],start[1:4];b,q=end[0],end[1:4];duration=float(np.linalg.norm(np.asarray(q)-p))/env.config.process.travel_speed_mm_s
 if duration>b-a+1e-9:return None,'direct_travel_exceeds_gap'
 direct=parts([[0.,*p,'T'],[duration,*q,'W']],env.bases[r,:2],step) if duration>1e-10 else ([],[],np.empty((0,2)))
 lower=a;upper=b-duration;blocked=[];motions=[];proof=[];c=env.config.collision
 for other in range(3):
  if other==r:continue
  rr=window(rows[other],a,b);g=all_parts(rr,env.bases[other,:2],step);trees=(STRtree(g[0]),STRtree(g[1]));proof.append((other,g,trees))
  motions.extend((max(a,x[0]),min(b,y[0])) for x,y in zip(rr,rr[1:]) if x[4]!='W')
  for kind,point in (('previous',p),('next',q)):
   shapes=point_shapes(point,env.bases[r,:2]);pairs=hits(([shapes[0]],[shapes[1]]),trees,thresholds[r,other],c)
   if len(pairs):
    tt=g[2][np.unique(pairs[:,1])];tt=tt[(tt[:,1]>a+1e-9)&(tt[:,0]<b-1e-9)]
    if len(tt):
     if kind=='previous':upper=min(upper,max(a,float(tt[:,0].min())-GUARD))
     else:lower=max(lower,float(tt[:,1].max())+GUARD-duration)
  if duration>1e-10:
   hh=hits(direct,trees,thresholds[r,other],c)
   if len(hh):blocked.extend(forbidden(direct[2],g[2],hh).tolist())
 intervals=allowed_intervals(blocked,lower,upper);depart=best_departure(intervals,duration,motions)
 if depart is None:return None,'no_safe_finite_wait_and_travel_window'
 own=gap_rows(a,b,p,q,depart,duration,end[4]);g=all_parts(own,env.bases[r,:2],step)
 # Independent final certificate covers finite waits, including parked homes.
 for other,old,trees in proof:
  hh=hits(g,trees,thresholds[r,other],c)
  if len(hh):
   x,y=g[2][hh[:,0]],old[2][hh[:,1]]
   if np.any(np.minimum(x[:,1],y[:,1])-np.maximum(x[:,0],y[:,0])>1e-8):return None,'final_timed_certificate_rejected'
 return own,dict(depart_s=depart,arrive_s=depart+duration,wait_previous_s=depart-a,wait_next_s=b-depart-duration,direct_travel_s=duration,certified_interval=[a,b])

def propose_advance(env,r,start,body,barrier,rows,thresholds,step):
 # Shift only the next layer body. Reconnect at its original home time.
 a=start[0];p=start[1:4];b=body[0][0];q=body[0][1:4];end=body[-1][0]
 d=float(np.linalg.norm(np.asarray(q)-p))/env.config.process.travel_speed_mm_s
 pattern=[[0.,*p,'T']]
 if d>1e-10:pattern.append([d,*q,body[0][4]])
 else:pattern[0][4]=body[0][4]
 pattern.extend([[v[0]-b+d,*v[1:]] for v in body[1:]]);pattern[-1][4]='W'
 g=all_parts(pattern,env.bases[r,:2],step);duration=pattern[-1][0];lower=max(a,barrier-d);upper=b-d;blocked=[];proof=[];c=env.config.collision
 for other in range(3):
  if other==r:continue
  old=all_parts(window(rows[other],a,end),env.bases[other,:2],step);trees=(STRtree(old[0]),STRtree(old[1]));proof.append((other,old,trees))
  for kind,point in (('previous',p),('home',body[-1][1:4])):
   shapes=point_shapes(point,env.bases[r,:2]);hh=hits(([shapes[0]],[shapes[1]]),trees,thresholds[r,other],c)
   if len(hh):
    tt=old[2][np.unique(hh[:,1])];tt=tt[(tt[:,1]>a+1e-9)&(tt[:,0]<end-1e-9)]
    if len(tt):
     if kind=='previous':upper=min(upper,max(a,float(tt[:,0].min())-GUARD))
     else:lower=max(lower,float(tt[:,1].max())+GUARD-duration)
  hh=hits(g,trees,thresholds[r,other],c)
  if len(hh):blocked.extend(forbidden(g[2],old[2],hh).tolist())
 intervals=allowed_intervals(blocked,lower,upper)
 if not intervals:return None
 depart=intervals[0][0];own=[[v[0]+depart,*v[1:]] for v in pattern]
 if depart>a+1e-9:own.insert(0,[a,*p,'W'])
 if own[-1][0]<end-1e-9:own.append([end,*body[-1][1:4],body[-1][4]])
 else:own[-1][4]=body[-1][4]
 gg=all_parts(own,env.bases[r,:2],step)
 for other,old,trees in proof:
  hh=hits(gg,trees,thresholds[r,other],c)
  if len(hh):
   x,y=gg[2][hh[:,0]],old[2][hh[:,1]]
   if np.any(np.minimum(x[:,1],y[:,1])-np.maximum(x[:,0],y[:,0])>1e-8):return None
 return own,dict(kind='advance_next_layer',depart_s=depart,arrive_s=depart+d,wait_previous_s=depart-a,advance_s=b-depart-d,direct_travel_s=d,certified_interval=[a,end],previous_layer_barrier_s=barrier)

def compress_global_idle(rows):
 motions=merge((a[0],b[0]) for rr in rows for a,b in zip(rr,rr[1:]) if a[4]!='W');end=max((b for a,b in motions),default=0.)
 holes=[];last=0.
 for a,b in motions:
  if a>last+1e-9:holes.append((last,a))
  last=b
 rights=np.asarray([b for a,b in holes]);lefts=np.asarray([a for a,b in holes]);cumulative=np.r_[0.,np.cumsum(rights-lefts)]
 def warp(t):
  t=min(t,end);n=np.searchsorted(rights,t,side='right');cut=float(cumulative[n])
  if n<len(holes) and t>lefts[n]:cut+=t-lefts[n]
  return float(t-cut)
 result=[];newend=warp(end)
 for rr in rows:
  own=[]
  for v in rr:
   item=[warp(v[0]),*v[1:]]
   if own and abs(item[0]-own[-1][0])<1e-8:
    if np.linalg.norm(np.asarray(item[1:4])-own[-1][1:4])>1e-5:raise ValueError('Global idle compression merged distinct positions')
    own[-1][4]=item[4]
   else:own.append(item)
  if own[-1][0]<newend-1e-9:own[-1][4]='W';own.append([newend,*own[-1][1:4],'W'])
  own[-1][4]='W';result.append(own)
 return result,dict(makespan_s=newend,removed_global_idle_s=float(sum(b-a for a,b in holes)),removed_intervals=holes)

def schedule_finite_stay(env,oldrows,step=10.):
 rows=[[v.copy() for v in rr] for rr in oldrows];thresholds=full_thresholds(env);records=[]
 order=sorted(range(3),key=lambda r:(-max((b[0] for a,b in zip(rows[r],rows[r][1:]) if a[4]!='W'),default=0.),r))
 for r in order:
  rr=rows[r];other_ends={}
  for other in range(3):
   if other!=r:
    for x,y in zip(rows[other],rows[other][1:]):
     if x[4]=='D':other_ends[x[3]]=max(other_ends.get(x[3],0.),y[0])
  first=next((i for i in range(len(rr)-1) if rr[i][4]=='D'),len(rr))
  while first<len(rr)-1:
   second=next((j for j in range(first+1,len(rr)-1) if rr[j][4]=='D'),None)
   if second is None:break
   if abs(rr[first][3]-rr[second][3])<1e-6:first=second;continue
   i=first+1;j=second;segment=rr[i:j+1]
   if not any(np.linalg.norm(np.asarray(v[1:4])-env.homes[r])<1e-4 for v in segment):first=second;continue
   oldtravel=sum(b[0]-a[0] for a,b in zip(segment,segment[1:]) if a[4]=='T');direct=float(np.linalg.norm(np.asarray(rr[j][1:4])-rr[i][1:4]))/env.config.process.travel_speed_mm_s
   if oldtravel<=direct+1e-6:first=second;continue
   home=next((k for k in range(j+1,len(rr)) if np.linalg.norm(np.asarray(rr[k][1:4])-env.homes[r])<1e-4),None)
   advanced=None
   if home is not None and all(v[3]==rr[j][3] for v in rr[j:home] if v[4]=='D'):
    barrier=max([rr[i][0]]+[end for z,end in other_ends.items() if z<rr[j][3]])
    advanced=propose_advance(env,r,rr[i],rr[j:home+1],barrier,rows,thresholds,step)
   if advanced is not None:own,certificate=advanced;replace_end=home
   else:own,certificate=propose(env,r,rr[i],rr[j],rows,thresholds,step);replace_end=j
   record=dict(robot_id=r+1,from_z=rr[first][3],to_z=rr[second][3],original_travel_s=oldtravel,accepted=own is not None)
   if own is not None:
    rr[i:replace_end+1]=own;record.update(certificate,travel_saved_s=oldtravel-direct);first=i+next(k for k,v in enumerate(own) if v[4]=='D')
   else:record['rejected']=certificate;first=second
   records.append(record)
  print(f'Finite stay robot {r+1}: {sum(v["accepted"] for v in records if v["robot_id"]==r+1)} shortcuts',flush=True)
 compressed,details=compress_global_idle(rows);plain,plain_info=compress_global_idle(oldrows)
 selected='finite_stay' if details['makespan_s']<plain_info['makespan_s']-1e-6 else 'idle_only'
 return dict(rows=compressed if selected=='finite_stay' else plain,schedule=dict(kind=selected,shortcuts=records,robot_order=[r+1 for r in order],proof_step_mm=step,finite_stay_compressed=details,baseline_compressed=plain_info,makespan_s=min(details['makespan_s'],plain_info['makespan_s'])))
