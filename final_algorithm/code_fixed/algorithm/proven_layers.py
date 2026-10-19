"""Conservative spatial reservations, with unchanged final official validation.

The official collision model is XY arm capsules and TCP disks. Each route's
convex hull contains every interpolated TCP; adding the base contains every
arm centerline. Disjoint hulls with the official radii/clearance prove safety
for all times. Unproven pairs execute serially; no coarse time sampling.
"""
from itertools import permutations, combinations
import numpy as np
from shapely import MultiPoint
from .parallel_layers import nearest_route


def schedule_proven_layers(env):
    archive=[[[0.0,*h,'W']] for h in env.homes]; now=0.;trace=[]
    for z in sorted(set(t.start[2] for t in env.tasks)):
        groups=[[],[],[]]
        for i,t in enumerate(env.tasks):
            if t.start[2]!=z:continue
            eligible=[r for r in range(3) if all(np.linalg.norm(np.asarray(p[:2])-env.bases[r,:2]) <= env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in (t.start,t.end))]
            if not eligible:raise ValueError(f'Unreachable task {i}')
            mid=(np.asarray(t.start)+t.end)/2
            r=min(eligible,key=lambda r:(np.linalg.norm(mid[:2]-env.bases[r,:2]),r));groups[r].append(i)
        routes={r:nearest_route(env,g,r) for r,g in enumerate(groups) if g}
        def hull(r,active):
            rows=routes[r][0] if active and r in routes else [[0.,*env.homes[r],'W']]
            points=np.asarray([p[1:3] for p in rows],dtype=np.float32).astype(float)
            tcp=MultiPoint(points).convex_hull
            arm=MultiPoint(np.vstack((points,env.bases[r,:2]))).convex_hull
            return tcp,arm
        moving={r:hull(r,True) for r in range(3)}
        parked={r:hull(r,False) for r in range(3)}
        collision=env.config.collision
        def separated(a,b,ha,hb):
            tcp_required=env.robots[a].tcp_radius_mm+env.robots[b].tcp_radius_mm
            arm_required=env.robots[a].arm_envelope_radius_mm+env.robots[b].arm_envelope_radius_mm+collision.arm_clearance_mm
            eps=max(collision.geometry_epsilon_mm,1e-5)
            return ((not collision.check_tcp_radius or ha[0].distance(hb[0])>tcp_required+eps) and
                    (not collision.check_arm_envelope or ha[1].distance(hb[1])>arm_required+eps))
        for a,b in combinations(range(3),2):
            for ha,hb in ((moving[a],parked[b]),(parked[a],moving[b]),(parked[a],parked[b])):
                if not separated(a,b,ha,hb):
                    raise RuntimeError(f'Spatial reservation cannot prove route/home separation: R{a+1}, R{b+1}, z={z}')
        conflicts={tuple(sorted((a,b))) for a,b in combinations(routes,2) if not separated(a,b,moving[a],moving[b])}
        duration={r:routes[r][0][-1][0] for r in routes};candidates=[]
        for order in permutations(routes):
            starts={}
            for r in order:
                starts[r]=max([0.]+[starts[k]+duration[k] for k in starts if tuple(sorted((r,k))) in conflicts])
            candidates.append((max(starts[r]+duration[r] for r in starts),tuple(order),starts))
        layer_time,_,starts=min(candidates,key=lambda x:(x[0],x[1]))
        for r in range(3):
            if r in routes:
                local=[[row[0]+starts[r],*row[1:]] for row in routes[r][0]]
                if starts[r]>0:local.insert(0,[0.,*env.homes[r],'W'])
                if local[-1][0]<layer_time:local.append([layer_time,*env.homes[r],'W'])
            else:local=[[0.,*env.homes[r],'W'],[layer_time,*env.homes[r],'W']]
            own=archive[r]
            if now>own[-1][0]:own.append([now,*env.homes[r],'W'])
            own[-1][4]=local[0][4];own.extend([[row[0]+now,*row[1:]] for row in local[1:]])
            if r in routes:
                trace.extend(dict(task=t['task'],robot_id=r+1,makespan_s=now+starts[r]+t['finish_local_s']) for t in routes[r][1])
        now+=layer_time
        print(f'Proven layer z={z}: tasks={sum(map(len,groups))}, starts={starts}, conflicts={sorted(conflicts)}, makespan={now:.3f}',flush=True)
    env.rows,env.index,env.makespan,env.done=archive,len(env.tasks),now,True
    return trace
