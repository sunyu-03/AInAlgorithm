"""Reuse per-instance successful repair rules; re-prove every repaired layer.
No model identifier is used by this algorithm. Candidate evaluation cap is fixed.
"""
from pathlib import Path
import json,time
from itertools import combinations
import numpy as np
from fixed_boundary_lns import records,rows_clip,patch,directed,order_tasks,route,joint_schedule
from resident_reassign import full_thresholds

def template_groups(env,items,rule):
 pair=rule['pair'];size=rule['bundle_size'];selected=sorted([t for t in items if t['robot'] in pair],key=lambda t:(-t['end'],t['robot']))[:size]
 if not selected:return None
 ids={id(t) for t in selected};groups={r:[t for t in items if t['robot']==r and id(t) not in ids] for r in range(3)}
 axis=np.asarray(rule['axis']);projection=np.array([(t['p'][:2]+t['q'][:2])/2 for t in selected])@axis;threshold=float(np.quantile(projection,rule['quantile']))
 for t,value in zip(selected,projection):
  preferred=pair[int((value>threshold)^rule['flip'])];other=pair[0] if preferred==pair[1] else pair[1]
  allowed=[r for r in [preferred,other] if all(np.linalg.norm(p[:2]-env.bases[r,:2])<=env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in [t['p'],t['q']])]
  if not allowed:return None
  groups[allowed[0]].append(t)
 return groups

def rules_from_seed(seed):
 records=json.loads(Path(seed).read_text())['layers'];rules=[];seen=set()
 for r in sorted([r for r in records if r.get('accepted') and r['change']['kind']=='spatial_joint'],key=lambda x:(-x['saved_s'],x['z'])):
  c=r['change'];key=json.dumps({k:c[k] for k in ['pair','bundle_size','quantile','flip','axis']},sort_keys=True)
  if key not in seen:rules.append(c);seen.add(key)
  if len(rules)==3:break
 return rules

def optimize(env,rows,budget_s=1200.,max_layers=12,step=10.,seed_schedule=None):
 started=time.perf_counter();expected=directed(rows);thresholds=full_thresholds(env);rules=rules_from_seed(seed_schedule);tasks=records(rows);grouped={}
 for t in tasks:grouped.setdefault(t['z'],[]).append(t)
 cumulative=0.;details=[];evaluations=0;stopped='all_layers';limit=1200
 for z,original in sorted(grouped.items()):
  if not rules:stopped='no_successful_spatial_rule';break
  if evaluations>=limit:stopped='evaluation_cap';break
  if time.perf_counter()-started>budget_s:stopped='time_guard';break
  tt=[dict(t,start=t['start']-cumulative,end=t['end']-cumulative) for t in original];b=max(t['end'] for t in tt);best=None;rejected={}
  for rule in rules:
   if evaluations>=limit or time.perf_counter()-started>budget_s:break
   selected=sorted([t for t in tt if t['robot'] in rule['pair']],key=lambda t:(-t['end'],t['robot']))[:rule['bundle_size']]
   if not selected:continue
   a=min(t['start'] for t in selected)
   while True:
    crossing=[t for t in tt if t['start']<a-1e-8 and t['end']>a+1e-8]
    if not crossing:break
    a=min(t['start'] for t in crossing)
   items=[t for t in tt if t['start']>=a-1e-8 and t['end']<=b+1e-8];groups=template_groups(env,items,rule)
   if groups is None:rejected['unreachable']=rejected.get('unreachable',0)+1;continue
   clips=[rows_clip(rr,a,b) for rr in rows];routes=[]
   for r in range(3):
    order=order_tasks(groups[r],np.asarray(clips[r][0][1:4]),np.asarray(clips[r][-1][1:4]));routes.append(route(env,r,order,np.asarray(clips[r][0][1:4]),np.asarray(clips[r][-1][1:4])))
   result,certificate=joint_schedule(env,routes,b-a,step,thresholds);evaluations+=1
   if result is None:rejected[certificate]=rejected.get(certificate,0)+1;continue
   local,info=result;gain=b-a-info['makespan_s']
   if gain>1e-4 and (best is None or gain>best[0]):best=(gain,local,a,rule,certificate,info,len(items))
  rec=dict(z=z,accepted=best is not None,rejections=rejected)
  if best:
   gain,local,a,rule,certificate,info,count=best;rows=patch(rows,local,a,b,b-a-gain);cumulative+=gain
   rec.update(saved_s=gain,window=[a,b],window_tasks=count,change=rule,reservation=certificate)
  details.append(rec)
  if best or len(details)%20==0:print(f'Sweep layer {z}: {rec.get("saved_s",0.):.6f}s; cumulative {cumulative:.6f}s; evaluations {evaluations}',flush=True)
 assert directed(rows)==expected,'Deposition changed during sweep'
 assert all(all(y[0]>x[0]+1e-9 for x,y in zip(rr,rr[1:])) for rr in rows),'Nonincreasing timestamp'
 return dict(rows=rows,schedule=dict(kind='fixed_boundary_template_sweep',seed_schedule=str(seed_schedule),rules=rules,layers=details,accepted_count=sum(v['accepted'] for v in details),saved_s=cumulative,search_s=time.perf_counter()-started,budget_s=budget_s,evaluation_cap=limit,evaluations=evaluations,stopped=stopped,proof_step_mm=step,makespan_s=max(rr[-1][0] for rr in rows),deposition_direction_unchanged=True))

