"""Fixed-position window repair; preserve deposition geometry; direction may change.
Outside a repaired window only a common suffix time translation is permitted.
"""
from itertools import permutations,combinations
from collections import Counter
import time
import numpy as np
from shapely import STRtree
from parallel_scheduler import parts
from resident_reassign import full_thresholds,point_shapes
from flexible_parallel import earliest_intervals
from finite_stay import compress_global_idle
GUARD=1e-6

def rows_clip(rr,a,b):
 def at(t):
  ts=np.asarray([r[0] for r in rr]);j=max(0,min(len(rr)-1,int(np.searchsorted(ts,t,side='right'))-1))
  x=rr[j]
  if abs(x[0]-t)<1e-8:return [t,*x[1:]]
  if j==len(rr)-1:return [t,*x[1:]]
  y=rr[j+1]
  if abs(y[0]-t)<1e-8:return [t,*y[1:]]
  if x[4]=='D':raise ValueError('Window cuts a deposition segment')
  p=np.asarray(x[1:4])+(np.asarray(y[1:4])-x[1:4])*(t-x[0])/(y[0]-x[0])
  return [t,*p,x[4]]
 return [at(a)]+[list(r) for r in rr if a+1e-8<r[0]<b-1e-8]+[at(b)]

def patch(rows,local,a,b,new_duration):
 delta=b-a-new_duration;result=[]
 for r,rr in enumerate(rows):
  middle=[[a+v[0],*v[1:]] for v in local[r]]
  assert np.linalg.norm(np.asarray(middle[0][1:4])-rows_clip(rr,a,b)[0][1:4])<1e-6
  assert np.linalg.norm(np.asarray(middle[-1][1:4])-rows_clip(rr,a,b)[-1][1:4])<1e-6
  middle[-1][4]=rows_clip(rr,a,b)[-1][4]
  result.append([list(v) for v in rr if v[0]<a-1e-8]+middle+[[v[0]-delta,*v[1:]] for v in rr if v[0]>b+1e-8])
 return result

def records(rows):
 ans=[]
 for r,rr in enumerate(rows):
  for x,y in zip(rr,rr[1:]):
   if x[4]=='D':ans.append(dict(robot=r,start=x[0],end=y[0],p=np.asarray(x[1:4]),q=np.asarray(y[1:4]),z=x[3]))
 return ans

def directed(rows):
 return Counter((tuple(round(float(x),6) for x in a[1:4]),tuple(round(float(x),6) for x in b[1:4])) for rr in rows for a,b in zip(rr,rr[1:]) if a[4]=='D')

def undirected(rows):
 return Counter(tuple(sorted((tuple(round(float(x),6) for x in a[1:4]),tuple(round(float(x),6) for x in b[1:4])))) for rr in rows for a,b in zip(rr,rr[1:]) if a[4]=='D')

def order_tasks(tasks,start,end):
 if not tasks:return []
 def orient(seq):
  pp=np.array([[t['p'],t['q']] for t in seq]);qq=pp[:,::-1,:];cost=np.linalg.norm(pp[0]-start,axis=1);parents=[]
  for i in range(1,len(seq)):
   edge=cost[:,None]+np.linalg.norm(qq[i-1][:,None,:]-pp[i][None,:,:],axis=2);parent=np.argmin(edge,axis=0);cost=edge[parent,np.arange(2)];parents.append(parent)
  state=int(np.argmin(cost+np.linalg.norm(qq[-1]-end,axis=1)));states=[0]*len(seq)
  for i in range(len(seq)-1,-1,-1):
   states[i]=state
   if i:state=int(parents[i-1][state])
  return [dict(t,p=t['q'],q=t['p']) if reverse else t for t,reverse in zip(seq,states)]
 def cost(seq):return float(np.linalg.norm(seq[0]['p']-start)+sum(np.linalg.norm(a['q']-b['p']) for a,b in zip(seq,seq[1:]))+np.linalg.norm(seq[-1]['q']-end))
 def improve(seq):
  seq=orient(seq);n=len(seq)
  for _ in range(12):
   pp=np.array([t['p'] for t in seq]);qq=np.array([t['q'] for t in seq]);left=np.vstack([start,qq[:-1]]);right=np.vstack([pp[1:],end])
   change=np.linalg.norm(left[:,None,:]-qq[None,:,:],axis=2)+np.linalg.norm(pp[:,None,:]-right[None,:,:],axis=2)-np.linalg.norm(left-pp,axis=1)[:,None]-np.linalg.norm(qq-right,axis=1)[None,:]
   change[np.tril_indices(n,-1)]=np.inf;i,j=np.unravel_index(int(np.argmin(change)),change.shape)
   if change[i,j]>=-1e-7:break
   seq=seq[:i]+[dict(t,p=t['q'],q=t['p']) for t in seq[i:j+1][::-1]]+seq[j+1:];seq=orient(seq)
  return seq
 pending=set(range(len(tasks)));near=[];current=np.asarray(start)
 while pending:
  _,i,reverse=min((float(np.linalg.norm(current-tasks[i]['q' if reverse else 'p'])),i,reverse) for i in pending for reverse in [False,True])
  t=tasks[i];item=dict(t,p=t['q'],q=t['p']) if reverse else t;near.append(item);current=item['q'];pending.remove(i)
 return min([improve(list(tasks)),improve(near)],key=cost)

def route(env,r,tasks,start,end):
 rr=[[0.,*start,'W']];p=np.asarray(start)
 for t in tasks:
  for q,mode,speed in [(t['p'],'T',env.config.process.travel_speed_mm_s),(t['q'],'D',env.config.process.deposition_speed_mm_s)]:
   d=float(np.linalg.norm(q-p))
   if d>1e-9:rr[-1][4]=mode;rr.append([rr[-1][0]+d/speed,*q,'W']);p=q
 d=float(np.linalg.norm(end-p))
 if d>1e-9:rr[-1][4]='T';rr.append([rr[-1][0]+d/env.config.process.travel_speed_mm_s,*end,'W'])
 return rr

def geometry(env,r,rr,step):
 g=parts(rr,env.bases[r,:2],step);tcp=list(g[0]);arm=list(g[1]);count=len(tcp)
 for p in [rr[0][1:4],rr[-1][1:4]]:
  x=point_shapes(p,env.bases[r,:2]);tcp.append(x[0]);arm.append(x[1])
 return (tcp,arm,np.asarray(g[2]).reshape(-1,2),count)

def joint_schedule(env,routes,H,step,thresholds):
 if any(rr[-1][0]>H-1e-5 for rr in routes):return None,'intrinsic_duration'
 gg=[geometry(env,r,rr,step) for r,rr in enumerate(routes)];trees=[(STRtree(g[0]),STRtree(g[1])) for g in gg];hits={}
 for r,q in permutations(range(3),2):
  found=[]
  for k,enabled in [(0,env.config.collision.check_tcp_radius),(1,env.config.collision.check_arm_envelope)]:
   if enabled:
    h=trees[q][k].query(gg[r][k],predicate='dwithin',distance=thresholds[r,q][k])
    if h.shape[1]:found.append(h.T)
  hits[r,q]=np.unique(np.concatenate(found),axis=0) if found else np.empty((0,2),int)
 def intervals(r,offset):
  duration=routes[r][-1][0]
  return np.vstack([gg[r][2]+offset,[0.,offset],[offset+duration,H]])
 best=None
 for priority in permutations(range(3)):
  offsets={};okay=True
  for r in priority:
   duration=routes[r][-1][0];lower=0.;upper=H-duration;blocked=[];count=gg[r][3]
   for q,offset in offsets.items():
    h=hits[r,q];old=intervals(q,offset)
    if not len(h):continue
    h=h[old[h[:,1],1]-old[h[:,1],0]>1e-9]
    for index,kind in [(count,'start'),(count+1,'end')]:
     z=h[h[:,0]==index]
     if len(z):
      tt=old[z[:,1]]
      if kind=='start':upper=min(upper,max(0.,float(tt[:,0].min())-GUARD))
      else:lower=max(lower,float(tt[:,1].max())+GUARD-duration)
    motion=h[h[:,0]<count]
    if len(motion):
     new=gg[r][2][motion[:,0]];past=old[motion[:,1]];blocked.extend(np.column_stack([past[:,0]-new[:,1]-GUARD,past[:,1]-new[:,0]+GUARD]).tolist())
   depart=earliest_intervals(blocked,lower)
   if depart>upper+1e-8:okay=False;break
   new=intervals(r,depart)
   for q,offset in offsets.items():
    h=hits[r,q]
    if len(h):
     x=new[h[:,0]];y=intervals(q,offset)[h[:,1]]
     active=(x[:,1]-x[:,0]>1e-9)&(y[:,1]-y[:,0]>1e-9)
     if np.any(active&(np.minimum(x[:,1],y[:,1])-np.maximum(x[:,0],y[:,0])>1e-8)):okay=False;break
   if not okay:break
   offsets[r]=depart
  if okay:
   finish=max(offsets[r]+routes[r][-1][0] for r in range(3));score=(finish,sum(offsets.values()),priority)
   if best is None or score<best[0]:best=(score,offsets)
 if best is None:return None,'collision_or_boundary_wait'
 result=[]
 for r,rr in enumerate(routes):
  off=best[1][r];own=[[v[0]+off,*v[1:]] for v in rr]
  if off>1e-9:own.insert(0,[0.,*rr[0][1:4],'W'])
  if own[-1][0]<H-1e-9:own[-1][4]='W';own.append([H,*rr[-1][1:4],'W'])
  else:own[-1][4]='W'
  # Stationary empty route still needs a positive-duration interval.
  if len(own)==1:own.append([H,*own[0][1:4],'W'])
  result.append(own)
 compressed,info=compress_global_idle(result)
 return (compressed,info),dict(priority=best[0][2],offsets=best[1])

def candidates(env,items,pair,size):
 groups={r:[t for t in items if t['robot']==r] for r in range(3)}
 selected=sorted([t for t in items if t['robot'] in pair],key=lambda t:(-t['end'],t['robot']))[:size]
 selected_ids={id(t) for t in selected};base={r:[t for t in groups[r] if id(t) not in selected_ids] for r in range(3)}
 yield groups,dict(kind='routing_only',pair=pair,bundle_size=size,selected_tasks=len(selected))
 if len(selected)<2:return
 mids=np.array([(t['p'][:2]+t['q'][:2])/2 for t in selected]);axes=[np.array([1.,0.]),np.array([0.,1.]),env.bases[pair[1],:2]-env.bases[pair[0],:2]]
 seen=set()
 for axis in axes:
  axis=axis/max(np.linalg.norm(axis),1e-10);projection=mids@axis
  for quantile in [.25,.5,.75]:
   threshold=float(np.quantile(projection,quantile))
   for flip in [False,True]:
    assignment=[];valid=True
    for t,value in zip(selected,projection):
     preferred=pair[int((value>threshold)^flip)];other=pair[0] if preferred==pair[1] else pair[1]
     allowed=[r for r in [preferred,other] if all(np.linalg.norm(p[:2]-env.bases[r,:2])<=env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in [t['p'],t['q']])]
     if not allowed:valid=False;break
     assignment.append(allowed[0])
    key=tuple(assignment)
    if not valid or key in seen:continue
    seen.add(key);result={r:list(base[r]) for r in range(3)}
    for t,r in zip(selected,assignment):result[r].append(t)
    yield result,dict(kind='spatial_joint',pair=pair,bundle_size=size,selected_tasks=len(selected),quantile=quantile,flip=flip,axis=axis.tolist())

def optimize(env,rows,budget_s=120.,max_layers=12,step=10.):
 start=time.perf_counter();expected=undirected(rows);thresholds=full_thresholds(env);log=[];accepted=[];tasks=records(rows)
 zs=sorted(set(t['z'] for t in tasks));scores=[]
 for z in zs:
  tt=[t for t in tasks if t['z']==z];lo=min(t['start'] for t in tt);hi=max(t['end'] for t in tt)
  loads=[sum(t['end']-t['start'] for t in tt if t['robot']==r) for r in range(3)]
  # Excess layer span is a bottleneck proxy; it is not claimed to be collision-only waiting.
  scores.append((-(hi-lo-max(loads)),z))
 for _,z in sorted(scores)[:max_layers]:
  if time.perf_counter()-start>budget_s:break
  tasks=records(rows);tt=[t for t in tasks if t['z']==z];layer_end=max(t['end'] for t in tt);best=None;trials=0;reasons={};window_seen=set()
  for size in [10,20,40]:
   for pair in combinations(range(3),2):
    if time.perf_counter()-start>budget_s:break
    select=sorted([t for t in tt if t['robot'] in pair],key=lambda t:-t['end'])[:size]
    if not select:continue
    a=min(t['start'] for t in select);b=layer_end
    # Expand to include complete D segments on every robot; travel may be split.
    while True:
     crossing=[t for t in tasks if t['start']<a-1e-8 and t['end']>a+1e-8]
     if not crossing:break
     a=min(t['start'] for t in crossing)
    key=(round(a,7),pair,size)
    if key in window_seen:continue
    window_seen.add(key);items=[t for t in tasks if t['start']>=a-1e-8 and t['end']<=b+1e-8]
    if not items or any(t['z']!=z for t in items):continue
    clips=[rows_clip(rr,a,b) for rr in rows];H=b-a;pool=[];signatures=set()
    for groups,change in candidates(env,items,pair,size):
     signature=tuple(tuple(sorted((t['start'],t['robot']) for t in groups[r])) for r in range(3))
     if signature in signatures:continue
     signatures.add(signature);routes=[]
     for r in range(3):
      order=order_tasks(groups[r],np.asarray(clips[r][0][1:4]),np.asarray(clips[r][-1][1:4]));routes.append(route(env,r,order,np.asarray(clips[r][0][1:4]),np.asarray(clips[r][-1][1:4])))
     intrinsic=max(rr[-1][0] for rr in routes)
     if intrinsic<H-1e-4:pool.append((intrinsic,sum(rr[-1][0] for rr in routes),change,routes))
     else:reasons['intrinsic_duration']=reasons.get('intrinsic_duration',0)+1
    for _,_,change,routes in sorted(pool,key=lambda v:v[:2])[:6]:
     if time.perf_counter()-start>budget_s:break
     result,detail=joint_schedule(env,routes,H,step,thresholds);trials+=1
     if result is None:reasons[detail]=reasons.get(detail,0)+1;continue
     local,info=result;gain=H-info['makespan_s']
     if gain>1e-4 and (best is None or gain>best[0]):best=(gain,local,a,b,change,detail,len(items),info)
  record=dict(z=z,proof_trials=trials,rejections=reasons)
  if best:
   gain,local,a,b,change,detail,count,info=best;new=patch(rows,local,a,b,b-a-gain)
   assert undirected(new)==expected,'Deposition changed during local repair'
   assert all(all(y[0]>x[0]+1e-9 for x,y in zip(rr,rr[1:])) for rr in new),'Nonincreasing timestamp'
   rows=new;record.update(accepted=True,saved_s=gain,window=[a,b],window_tasks=count,change=change,reservation=detail,removed_idle=info)
   accepted.append(record)
  else:record['accepted']=False
  log.append(record);print(f'Fixed boundary layer {z}: saved {record.get("saved_s",0.):.6f}s, trials {trials}',flush=True)
 return dict(rows=rows,schedule=dict(kind='fixed_boundary_spatial_lns',layers=log,accepted_count=len(accepted),saved_s=sum(v['saved_s'] for v in accepted),search_s=time.perf_counter()-start,budget_s=budget_s,bundle_sizes=[10,20,40],max_layers=max_layers,proof_step_mm=step,makespan_s=max(rr[-1][0] for rr in rows),deposition_direction_unchanged=False,deposition_geometry_unchanged=True,wait_score_is_bottleneck_proxy=True))
