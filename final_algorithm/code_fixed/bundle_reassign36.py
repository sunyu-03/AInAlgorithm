"""Bounded contiguous task bundles with the existing two-layer reservation engine."""
import numpy as np
from parallel_scheduler import eligible
from resident_reassign import full_thresholds,wait_scores
from two_layer_lookahead import Model,Engine,State,split_episodes,choose_lookahead,materialize
from joint_wait_search import evaluate_all

def endpoints(env,item):
 t=env.tasks[item[0]]
 return (np.asarray(t.end),np.asarray(t.start)) if item[1] else (np.asarray(t.start),np.asarray(t.end))

def travel(env,ss,r):
 p=env.homes[r];total=0.
 for item in ss:
  a,b=endpoints(env,item);total+=np.linalg.norm(a-p);p=b
 return float(total+np.linalg.norm(env.homes[r]-p))

def insert_bundle(env,ss,bundle,r):
 best=None;inside=sum(np.linalg.norm(endpoints(env,bundle[j+1])[0]-endpoints(env,bundle[j])[1]) for j in range(len(bundle)-1))
 for k in range(len(ss)+1):
  p=endpoints(env,ss[k-1])[1] if k else env.homes[r];q=endpoints(env,ss[k])[0] if k<len(ss) else env.homes[r]
  cost=float(np.linalg.norm(endpoints(env,bundle[0])[0]-p)+inside+np.linalg.norm(q-endpoints(env,bundle[-1])[1])-np.linalg.norm(q-p))
  if best is None or (cost,k)<best:best=(cost,k)
 return ss[:best[1]]+list(bundle)+ss[best[1]:],best[0]

def bundle_candidates(env,seq,wait,finish,z,sizes=(3,6),hot_limit=8):
 source=max(seq,key=lambda r:(finish.get((z,r),0.),r));ss=seq[source]
 duration={i:float(np.linalg.norm(np.asarray(env.tasks[i].end)-env.tasks[i].start))/env.config.process.deposition_speed_mm_s for i,f in ss}
 hot=sorted(range(len(ss)),key=lambda k:(-wait.get(ss[k][0],0.)-duration[ss[k][0]],ss[k][0]))[:hot_limit];result=[]
 for size in sizes:
  starts=sorted({max(0,min(k-j,len(ss)-size)) for k in hot for j in range(size)}) if len(ss)>=size else []
  best=None
  for k in starts:
   bundle=ss[k:k+size];rest=ss[:k]+ss[k+size:];saved=travel(env,ss,source)-travel(env,rest,source);urgency=sum(wait.get(i,0.)+duration[i] for i,f in bundle)
   for target in range(3):
    if target==source or not eligible(env,[i for i,f in bundle],target):continue
    ins,cost=insert_bundle(env,seq.get(target,[]),bundle,target)
    score=urgency+(saved-cost)/env.config.process.travel_speed_mm_s-max(0.,finish.get((z,target),0.)-finish.get((z,source),0.))
    rank=(-score,source,target,k)
    if best is None or rank<best[0]:best=(rank,target,bundle,rest,ins)
  if best:
   _,target,bundle,rest,ins=best;groups={r:list(v) for r,v in seq.items()};groups[source]=rest;groups[target]=ins
   result.append(({r:v for r,v in groups.items() if v},dict(kind='bundle_transfer',tasks=[i for i,f in bundle],from_robot=source+1,to_robot=target+1,bundle_size=size)))
 return result

def schedule_batch(env,sequences,oldrows,step=10.,beam=2):
 thresholds=full_thresholds(env);episodes,resident=split_episodes(env,sequences,oldrows);wait,finish=wait_scores(oldrows,sequences);zs=sorted(sequences)
 engine=Engine(env,thresholds);bases={};history=[];details=[];state=State()
 def base(index):
  if index not in bases:
   z=zs[index];bases[index]=Model(env,(index,0),z,episodes[z],dict(kind='preserved_assignment'),step,thresholds)
  return bases[index]
 for index,z in enumerate(zs):
  options=[base(index)];trials=[]
  for n,(seq,change) in enumerate(bundle_candidates(env,sequences[z],wait,finish,z),1):
   try:
    plan=evaluate_all(env,seq,step);options.append(Model(env,(index,n),z,plan['rows'],change,step,thresholds));trials.append(dict(change=change,home_layer_s=plan['end']))
   except ValueError as e:trials.append(dict(change=change,rejected=str(e)))
  chosen,rollouts=choose_lookahead(engine,state,options,base(index+1) if index+1<len(zs) else None,beam);state=chosen['state']
  history.append(dict(rows=chosen['model'].rows,offsets=dict(chosen['offsets'])))
  details.append(dict(z=z,option=chosen['model'].change,order=chosen['order'],offsets=chosen['offsets'],makespan_s=state.makespan,trials=trials,lookahead_trials=rollouts))
  print(f'Bundle layer {z}: {state.makespan:.3f}s; {chosen["model"].change["kind"]}',flush=True)
  engine.forget_before(index)
  for k in list(bases):
   if k<index:del bases[k]
 return dict(rows=materialize(env,history,state),schedule=dict(kind='bundle_two_layer',layers=details,makespan_s=state.makespan,bundle_sizes=[3,6],hot_limit=8,priority_beam=beam,proof_step_mm=step,resident_input_home_comparator=resident))
