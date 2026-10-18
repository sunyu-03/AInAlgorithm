"""Two-layer receding-horizon assignment/start reservation search.

Geometry and wait reservations are immutable. Only the current layer is
committed; the next unchanged assignment is a look-ahead rollout.
"""
from dataclasses import dataclass
from collections import OrderedDict
from itertools import combinations,permutations
import numpy as np
from shapely import STRtree,MultiPoint
from overlap_retimer import reserved_parts,forbidden
from flexible_parallel import earliest_intervals
from resident_reassign import full_thresholds,assignment_candidates,wait_scores
from joint_wait_search import evaluate_all
from task_swap_parallel import route

@dataclass(frozen=True)
class Reservation:
 model: object
 robot: int
 offset: float
 end: float

@dataclass(frozen=True)
class State:
 ends: tuple=(0.,0.,0.)
 barrier: float=0.
 reservations: tuple=()
 @property
 def makespan(self):return max(self.ends)

class Model:
 def __init__(self,env,uid,z,rows,change,step,thresholds):
  self.uid=uid;self.z=z;self.rows={};self.geometry={};self.trees={};self.first={};self.last={};self.change=change
  for r,rr in rows.items():
   moving=[i for i in range(len(rr)-1) if rr[i][4]!='W']
   if not moving:continue
   rr=[v.copy() for v in rr[moving[0]:moving[-1]+2]];origin=rr[0][0];rr=[[v[0]-origin,*v[1:]] for v in rr]
   if any(np.linalg.norm(np.asarray(v[1:4])-env.homes[r])>1e-4 for v in (rr[0],rr[-1])):raise ValueError('Layer route needs home endpoints')
   dd=[(a[0],b[0]) for a,b in zip(rr,rr[1:]) if a[4]=='D']
   if not dd:raise ValueError('Moving layer has no deposition')
   g=reserved_parts(rr,env.bases[r,:2],env.homes[r],step);trees=(STRtree(g[0]),STRtree(g[1]))
   for other in range(3):
    if r==other:continue
    p=np.asarray(env.homes[other,:2],np.float32).astype(float);park=(MultiPoint([p]).convex_hull,MultiPoint([p,env.bases[other,:2]]).convex_hull)
    c=env.config.collision
    for k,enabled in ((0,c.check_tcp_radius),(1,c.check_arm_envelope)):
     if enabled and len(trees[k].query_nearest(park[k],max_distance=thresholds[r,other][k],all_matches=False)):raise ValueError('Parked-home proof failed')
   self.rows[r]=rr;self.geometry[r]=g;self.trees[r]=trees;self.first[r]=dd[0][0];self.last[r]=dd[-1][1]
  if not self.rows:raise ValueError('Empty layer model')

class Engine:
 def __init__(self,env,thresholds,cache_bytes=64*1024*1024):
  self.env=env;self.thresholds=thresholds;self.cache=OrderedDict();self.bytes=0;self.limit=cache_bytes;self.transitions=0
 def pairs(self,a,r,b,q,cutoff=None):
  times=b.geometry[q][2];ids=np.flatnonzero(times[:,1]>=cutoff-1e-6) if cutoff is not None else None
  if ids is not None and not len(ids):return np.empty((0,2),np.uint32)
  if ids is not None and len(ids)==len(times):ids=None
  key=(a.uid,r,b.uid,q,None if ids is None else tuple(ids))
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  found=[];c=self.env.config.collision;count=len(times)
  for k,enabled in ((0,c.check_tcp_radius),(1,c.check_arm_envelope)):
   if not enabled:continue
   if ids is None:
    hit=b.trees[q][k].query(a.geometry[r][k],predicate='dwithin',distance=self.thresholds[r,q][k]);left,right=hit
   else:
    hit=a.trees[r][k].query([b.geometry[q][k][i] for i in ids],predicate='dwithin',distance=self.thresholds[r,q][k]);left,right=hit[1],ids[hit[0]]
   if len(left):found.append(left.astype(np.int64)*count+right)
  codes=np.unique(np.concatenate(found)) if found else np.empty(0,np.int64)
  arr=np.column_stack((codes//count,codes%count)).astype(np.uint32)
  if arr.nbytes<=self.limit:
   while self.cache and self.bytes+arr.nbytes>self.limit:
    _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
   self.cache[key]=arr;self.bytes+=arr.nbytes
  return arr
 def forget_before(self,index):
  for k in list(self.cache):
   if k[0][0]<index or k[2][0]<index:self.bytes-=self.cache.pop(k).nbytes
 def transitions_for(self,state,model):
  self.transitions+=1;lowers={r:max(0.,state.ends[r],state.barrier-model.first[r]) for r in model.rows};lower=min(lowers.values())
  retained=tuple(v for v in state.reservations if v.end>=lower-1e-6);base={}
  for r in model.rows:
   blocked=[]
   for old in retained:
    if r==old.robot:continue
    hit=self.pairs(model,r,old.model,old.robot,lowers[r]-old.offset)
    if len(hit):blocked.append(forbidden(model.geometry[r][2],old.model.geometry[old.robot][2]+old.offset,hit))
   base[r]=blocked
  internal={(r,q):self.pairs(model,r,model,q) for r in model.rows for q in model.rows if r!=q};results=[]
  for order in permutations(sorted(model.rows)):
   offsets={}
   for r in order:
    blocked=list(base[r])
    for q,off in offsets.items():
     hit=internal[r,q]
     if len(hit):blocked.append(forbidden(model.geometry[r][2],model.geometry[q][2]+off,hit))
    offsets[r]=earliest_intervals(np.concatenate(blocked) if blocked else [],lowers[r])
   ends=list(state.ends);reservations=list(retained)
   for r,off in offsets.items():
    ends[r]=off+model.rows[r][-1][0];reservations.append(Reservation(model,r,off,ends[r]))
   barrier=max(offsets[r]+model.last[r] for r in model.rows)
   nxt=State(tuple(ends),barrier,tuple(reservations));results.append(dict(state=nxt,model=model,offsets=offsets,order=order))
  return sorted(results,key=local_score)

def local_score(v):return (v['state'].makespan,v['state'].barrier,sum(v['state'].ends),v['model'].uid,v['order'])

def choose_lookahead(engine,state,options,next_base,beam=2):
 trials=[];winner=None
 for model in options:
  for current in engine.transitions_for(state,model)[:beam]:
   future=min(engine.transitions_for(current['state'],next_base),key=local_score) if next_base else current
   score=(future['state'].makespan,future['state'].barrier,*local_score(current))
   trials.append(dict(option=model.change,order=current['order'],current_end_s=current['state'].makespan,projected_end_s=future['state'].makespan,projected_barrier_s=future['state'].barrier,next_order=future['order'] if next_base else None))
   if winner is None or score<winner[0]:winner=(score,current)
 return winner[1],trials

def split_episodes(env,sequences,oldrows):
 result={z:{} for z in sequences};resident=False
 for r,rr in oldrows.items():
  i=0
  while i<len(rr)-1:
   a=rr[i]
   if a[4]=='W' or np.linalg.norm(np.asarray(a[1:4])-env.homes[r])>1e-4:i+=1;continue
   j=i+1
   while j<len(rr) and np.linalg.norm(np.asarray(rr[j][1:4])-env.homes[r])>1e-4:j+=1
   if j==len(rr):raise ValueError('Unterminated home episode')
   zs={v[3] for v in rr[i:j] if v[4]=='D'}
   if len(zs)>1:resident=True;break
   if len(zs)==1:
    z=next(iter(zs))
    if r in result[z]:raise ValueError('Multiple home episodes in one layer')
    result[z][r]=[v.copy() for v in rr[i:j+1]]
   i=j
  if resident:break
 if resident:
  # Generic comparison contract for input without home layer boundaries.
  result={z:{r:route(env,ss,r)[0] for r,ss in groups.items()} for z,groups in sequences.items()}
 if any(set(result[z])!=set(g) for z,g in sequences.items()):raise ValueError('Incomplete episode reconstruction')
 return result,resident

def materialize(env,history,state):
 archive=[[[0.,*h,'W']] for h in env.homes]
 for entry in history:
  for r,rr in entry['rows'].items():
   off=entry['offsets'][r]
   if off<archive[r][-1][0]-1e-7:raise ValueError('Own-route overlap')
   if off>archive[r][-1][0]+1e-9:archive[r][-1][4]='W';archive[r].append([off,*env.homes[r],'W'])
   archive[r][-1][4]=rr[0][4];archive[r].extend([[v[0]+off,*v[1:]] for v in rr[1:]])
 end=state.makespan
 for r in range(3):
  if archive[r][-1][0]<end:archive[r][-1][4]='W';archive[r].append([end,*env.homes[r],'W'])
 return archive

def schedule_both(env,sequences,oldrows,step=10.,beam=2):
 thresholds=full_thresholds(env);episodes,resident=split_episodes(env,sequences,oldrows);wait,finish=wait_scores(oldrows,sequences);zs=sorted(sequences)
 engine=Engine(env,thresholds);bases={};history={'myopic':[],'lookahead':[]};states={k:State() for k in history};details={k:[] for k in history};construction=[]
 def base(index):
  if index not in bases:
   z=zs[index];bases[index]=Model(env,(index,0),z,episodes[z],dict(kind='preserved_assignment'),step,thresholds)
  return bases[index]
 for index,z in enumerate(zs):
  models=[base(index)];records=[]
  for n,(seq,change) in enumerate(assignment_candidates(env,sequences[z],wait,finish,z),1):
   try:
    plan=evaluate_all(env,seq,step);model=Model(env,(index,n),z,plan['rows'],change,step,thresholds);models.append(model);records.append(dict(change=change,home_layer_s=plan['end']))
   except ValueError as e:records.append(dict(change=change,rejected=str(e)))
  construction.append(dict(z=z,candidates=records));future=base(index+1) if index+1<len(zs) else None
  myopic=min((v for m in models for v in engine.transitions_for(states['myopic'],m)),key=local_score)
  chosen,trials=choose_lookahead(engine,states['lookahead'],models,future,beam)
  for policy,entry in [('myopic',myopic),('lookahead',chosen)]:
   states[policy]=entry['state'];history[policy].append(dict(rows=entry['model'].rows,offsets=dict(entry['offsets'])))
   details[policy].append(dict(z=z,option=entry['model'].change,order=entry['order'],offsets=entry['offsets'],makespan_s=entry['state'].makespan,last_deposition_s=entry['state'].barrier,lookahead_trials=trials if policy=='lookahead' else []))
  print(f'Lookahead layer {z}: one_layer={states["myopic"].makespan:.3f}s, two_layer={states["lookahead"].makespan:.3f}s',flush=True)
  # Keep pair memory bounded; active State reservations retain needed geometry.
  engine.forget_before(index)
  for k in list(bases):
   if k<index:del bases[k]
 outputs={k:dict(rows=materialize(env,history[k],states[k]),schedule=dict(kind=k,layers=details[k],makespan_s=states[k].makespan,construction=construction,horizon_layers=2 if k=='lookahead' else 1,priority_beam=beam if k=='lookahead' else None,proof_step_mm=step,resident_input_home_comparator=resident,transition_calls=engine.transitions)) for k in history}
 return outputs
