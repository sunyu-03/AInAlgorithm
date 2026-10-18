"""Whole-layer reservations with official collision checks and serial fallback."""
from itertools import permutations
import numpy as np
from waam_validator import validate_trajectory_set, compute_reach_metrics
from waam_validator.collision.simulator import run_collision_analysis


def nearest_route(env, indices, robot):
    current=np.asarray(env.homes[robot],float)
    rows=[[0.0,*current,'W']]; trace=[]
    starts=np.asarray([env.tasks[i].start for i in indices],float)
    ends=np.asarray([env.tasks[i].end for i in indices],float)
    pending=np.ones(len(indices),bool)
    while pending.any():
        positions=np.flatnonzero(pending)
        points=np.concatenate((starts[positions],ends[positions]))
        other=np.concatenate((ends[positions],starts[positions]))
        idx=np.tile(positions,2)
        distance=np.linalg.norm(points-current,axis=1)
        choice=int(np.lexsort((points[:,1],points[:,0],np.asarray(indices)[idx],distance))[0])
        for point,mode,speed in ((points[choice],'T',env.config.process.travel_speed_mm_s),
                                  (other[choice],'D',env.config.process.deposition_speed_mm_s)):
            d=float(np.linalg.norm(point-np.asarray(rows[-1][1:4])))
            if d>1e-9:
                rows[-1][4]=mode; rows.append([rows[-1][0]+d/speed,*point,'W'])
        position=int(idx[choice]); pending[position]=False; current=other[choice]
        trace.append(dict(task=int(indices[position]),robot_id=robot+1,finish_local_s=rows[-1][0]))
    d=float(np.linalg.norm(current-env.homes[robot]))
    if d>1e-9:
        rows[-1][4]='T'; rows.append([rows[-1][0]+d/env.config.process.travel_speed_mm_s,*env.homes[robot],'W'])
    return rows,trace


def schedule_parallel_layers(env):
    archive=[[[0.0,*h,'W']] for h in env.homes]; now=0.0; trace=[]
    for z in sorted(set(t.start[2] for t in env.tasks)):
        indices=[i for i,t in enumerate(env.tasks) if t.start[2]==z]
        assigned=[[],[],[]]
        for i in indices:
            t=env.tasks[i]; midpoint=(np.asarray(t.start)+t.end)/2
            eligible=[r for r in range(3) if all(np.linalg.norm(np.asarray(p[:2])-env.bases[r,:2]) <= env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in (t.start,t.end))]
            if not eligible: raise ValueError(f'Unreachable task {i}')
            robot=min(eligible,key=lambda r:(np.linalg.norm(midpoint[:2]-env.bases[r,:2]),r))
            assigned[robot].append(i)
        routes={r:nearest_route(env,group,r) for r,group in enumerate(assigned) if group}
        duration={r:routes[r][0][-1][0] for r in routes}; memo={}
        def build(starts):
            end=max((starts[r]+duration[r] for r in starts),default=1e-6)
            result=[]
            for r in range(3):
                if r not in starts:
                    result.append([[0.0,*env.homes[r],'W'],[end,*env.homes[r],'W']]);continue
                offset=starts[r]; own=[[row[0]+offset,*row[1:]] for row in routes[r][0]]
                if offset>0:own.insert(0,[0.0,*env.homes[r],'W'])
                if own[-1][0]<end:own.append([end,*env.homes[r],'W'])
                result.append(own)
            return result
        def safe(starts):
            key=tuple(sorted(starts.items()))
            if key not in memo:
                tr=env._trajectory(build(starts))
                memo[key]=(not validate_trajectory_set(tr,env.config).violations and
                    compute_reach_metrics(tr,env.config).passed and not run_collision_analysis(tr,env.config).events)
            return memo[key]
        candidates=[]
        for order in permutations(routes):
            starts={}
            for r in order:
                options=sorted(set([0.0]+[starts[k]+duration[k] for k in starts]))
                for offset in options:
                    trial={**starts,r:offset}
                    if safe(trial):starts=trial;break
                else:break
            if len(starts)==len(routes):
                candidates.append((max(starts[r]+duration[r] for r in starts),tuple(order),starts))
        if not candidates:
            raise RuntimeError(f'No safe whole-layer reservations at z={z}; no unchecked schedule exported')
        layer_time,_,starts=min(candidates,key=lambda x:(x[0],x[1])); local=build(starts)
        for r in range(3):
            own=archive[r]
            if now>own[-1][0]:own.append([now,*env.homes[r],'W'])
            own[-1][4]=local[r][0][4]
            own.extend([[row[0]+now,*row[1:]] for row in local[r][1:]])
        for r in routes:
            trace.extend(dict(task=t['task'],robot_id=r+1,makespan_s=now+starts[r]+t['finish_local_s']) for t in routes[r][1])
        now+=layer_time
        print(f'Parallel layer z={z}: tasks={len(indices)}, starts={starts}, makespan={now:.3f}',flush=True)
    env.rows,env.index,env.makespan,env.done=archive,len(env.tasks),now,True
    return trace
