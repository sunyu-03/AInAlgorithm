"""Refine continuous collision reservations without changing deposition beads.

Proof-only motion subdivision plus route reversal search. All robots return
home between layers; all parked-home separation is proven independently.
"""
from itertools import combinations,permutations,product
import numpy as np
from shapely import STRtree
from algorithm.parallel_layers import nearest_route
from algorithm.proven_segments import merge_intervals
from parallel_scheduler import parts,home_safe,radii,eligible

def reverse_rows(rows):
 duration=rows[-1][0]
 return [[duration-rows[i][0],*rows[i][1:4],rows[i-1][4] if i else 'W'] for i in range(len(rows)-1,-1,-1)]

def reflected_times(times,duration):return duration-times[:,::-1]

def solve_offsets(routes,forbidden,durations):
 choices=[]
 for order in permutations(routes):
  starts={}
  for r in order:
   blocked=merge_intervals((starts[other]+a,starts[other]+b) for other in starts for a,b in forbidden[other,r])
   start=0.
   for left,right in blocked:
    if left<=start<=right:start=right+1e-6
   starts[r]=start
  choices.append((max(starts[r]+durations[r] for r in starts),tuple(order),starts))
 return min(choices,key=lambda v:v[:2])

def schedule(env,step=25.,reverse_search=True):
 layers={};archive=[[[0.,*h,'W']] for h in env.homes];now=0.;trace=[];diagnostics=[]
 for i,t in enumerate(env.tasks):layers.setdefault(t.start[2],[]).append(i)
 for z,indices in sorted(layers.items()):
  groups={}
  for i in indices:
   valid=[r for r in range(3) if eligible(env,[i],r)]
   if not valid:raise ValueError(f'Unreachable task {i}')
   mid=(np.asarray(env.tasks[i].start)+env.tasks[i].end)/2
   r=min(valid,key=lambda r:(np.linalg.norm(mid[:2]-env.bases[r,:2]),r));groups.setdefault(r,[]).append(i)
  routes={r:nearest_route(env,g,r) for r,g in groups.items()};durations={r:v[0][-1][0] for r,v in routes.items()}
  geometry={r:parts(v[0],env.bases[r,:2],step) for r,v in routes.items()}
  for r,g in geometry.items():
   if not home_safe(env,r,g):raise ValueError(f'Cannot prove parked-home separation at z={z}, robot={r+1}')
  pair_indices={}
  for a,b in combinations(sorted(routes),2):
   tcp,arm=radii(env,a,b);pairs=set();c=env.config.collision
   for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
    if enabled:
     found=STRtree(geometry[b][kind]).query(geometry[a][kind],predicate='dwithin',distance=radius)
     pairs.update(zip(found[0].tolist(),found[1].tolist()))
   pair_indices[a,b]=np.asarray(sorted(pairs),dtype=int).reshape(-1,2)
  robots=sorted(routes);variants=[]
  for flags in product((False,True) if reverse_search else (False,),repeat=len(robots)):
   reversal=dict(zip(robots,flags));times={r:reflected_times(geometry[r][2],durations[r]) if reversal[r] else geometry[r][2] for r in robots};forbidden={}
   for (a,b),pairs in pair_indices.items():
    intervals=merge_intervals((times[a][i,0]-times[b][j,1]-1e-6,times[a][i,1]-times[b][j,0]+1e-6) for i,j in pairs)
    forbidden[a,b]=intervals;forbidden[b,a]=[[-right,-left] for left,right in reversed(intervals)]
   end,order,starts=solve_offsets(robots,forbidden,durations)
   variants.append((end,flags,order,starts,forbidden))
  end,flags,order,starts,forbidden=min(variants,key=lambda v:v[:3]);reversal=dict(zip(robots,flags))
  for r in range(3):
   if r in routes:
    source=reverse_rows(routes[r][0]) if reversal[r] else routes[r][0]
    local=[[row[0]+starts[r],*row[1:]] for row in source]
    if starts[r]>0:local.insert(0,[0.,*env.homes[r],'W'])
    if local[-1][0]<end:local.append([end,*env.homes[r],'W'])
    for t in routes[r][1]:trace.append(dict(task=t['task'],robot_id=r+1,layer_z=z,reversed_route=reversal[r]))
   else:local=[[0.,*env.homes[r],'W'],[end,*env.homes[r],'W']]
   archive[r][-1][4]=local[0][4];archive[r].extend([[row[0]+now,*row[1:]] for row in local[1:]])
  diagnostics.append(dict(z=z,layer_start_s=now,duration_s=end,serial_s=sum(durations.values()),starts=starts,reversal=reversal,order=order,variants=len(variants),proof_step_mm=step))
  now+=end;print(f'Refined layer {z}: tasks={len(indices)}, layer_s={end:.3f}, total_s={now:.3f}, reversal={reversal}',flush=True)
 env.rows=archive;env.index=len(env.tasks);env.makespan=now;env.done=True
 return dict(trace=trace,layers=diagnostics,makespan_s=now,proof_step_mm=step,reverse_search=reverse_search)
