"""Continuous per-segment spatial proofs and forbidden departure intervals.

For an unseparated pair of swept segment hulls, execution time intervals must
not overlap. The arm hull contains every base-to-TCP segment for all times.
This conservative proof does not rely on sampled collision detection.
"""
from itertools import combinations, permutations
import numpy as np
from shapely import MultiPoint, STRtree
from .parallel_layers import nearest_route


def merge_intervals(intervals):
    result=[]
    for left,right in sorted(intervals):
        if result and left <= result[-1][1]: result[-1][1]=max(right,result[-1][1])
        else: result.append([float(left),float(right)])
    return result


def schedule_proven_segments(env):
    archive=[[[0.,*h,'W']] for h in env.homes]; now=0.; trace=[]; diagnostics=[]
    c=env.config.collision; guard=1e-6
    # Official interpolation casts every intermediate position to float32.
    # Eight ULP covers both XY rounding errors, including binade boundaries.
    maximum=max(np.max(np.abs(env.homes[:,:2])),
                max(abs(v) for t in env.tasks for p in (t.start,t.end) for v in p[:2]))
    eps=max(c.geometry_epsilon_mm,float(8*np.spacing(np.float32(maximum))),1e-5)
    layers={}
    for i,t in enumerate(env.tasks): layers.setdefault(t.start[2],[]).append(i)
    for z,indices in sorted(layers.items()):
        groups=[[],[],[]]
        for i in indices:
            t=env.tasks[i]
            eligible=[r for r in range(3) if all(np.linalg.norm(np.asarray(p[:2])-env.bases[r,:2]) <=
                      env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in (t.start,t.end))]
            if not eligible: raise ValueError(f'Unreachable task {i}')
            mid=(np.asarray(t.start)+t.end)/2
            groups[min(eligible,key=lambda r:(np.linalg.norm(mid[:2]-env.bases[r,:2]),r))].append(i)
        routes={r:nearest_route(env,g,r) for r,g in enumerate(groups) if g}
        duration={r:route[0][-1][0] for r,route in routes.items()}
        shapes={}; times={}
        for r in range(3):
            rows=routes[r][0] if r in routes else [[0.,*env.homes[r],'W'],[1e-6,*env.homes[r],'W']]
            points=np.asarray([row[1:3] for row in rows],np.float32).astype(float)
            shapes[r]=([MultiPoint(points[i:i+2]).convex_hull for i in range(len(points)-1)],
                       [MultiPoint(np.vstack((points[i:i+2],env.bases[r,:2]))).convex_hull for i in range(len(points)-1)])
            times[r]=np.asarray([row[0] for row in rows])
        forbidden={}
        for a,b in combinations(range(3),2):
            tcp=env.robots[a].tcp_radius_mm+env.robots[b].tcp_radius_mm+eps
            arm=env.robots[a].arm_envelope_radius_mm+env.robots[b].arm_envelope_radius_mm+c.arm_clearance_mm+eps
            # Always prove safety against other robots at their homes, including
            # delayed starts, completed routes, and idle robots.
            for moving,park in ((a,b),(b,a)):
                point=np.asarray(env.homes[park][:2],np.float32).astype(float)
                parked=(MultiPoint([point]).convex_hull,MultiPoint([point,env.bases[park,:2]]).convex_hull)
                if ((c.check_tcp_radius and any(g.distance(parked[0]) <= tcp for g in shapes[moving][0])) or
                    (c.check_arm_envelope and any(g.distance(parked[1]) <= arm for g in shapes[moving][1]))):
                    raise RuntimeError(f'Cannot prove segment/home separation: {a}, {b}, z={z}')
            pairs=set()
            for kind,enabled,radius in ((0,c.check_tcp_radius,tcp),(1,c.check_arm_envelope,arm)):
                if enabled:
                    found=STRtree(shapes[b][kind]).query(shapes[a][kind],predicate='dwithin',distance=radius)
                    pairs.update(zip(found[0].tolist(),found[1].tolist()))
            intervals=merge_intervals((times[a][i]-times[b][j+1]-guard,
                                      times[a][i+1]-times[b][j]+guard) for i,j in pairs)
            forbidden[a,b]=intervals
            forbidden[b,a]=[[-right,-left] for left,right in reversed(intervals)]
        candidates=[]
        for order in permutations(routes):
            starts={}
            for r in order:
                blocked=merge_intervals((starts[other]+left,starts[other]+right)
                                        for other in starts for left,right in forbidden[other,r])
                start=0.
                for left,right in blocked:
                    if left <= start <= right: start=right+guard
                starts[r]=start
            candidates.append((max(starts[r]+duration[r] for r in starts),tuple(order),starts))
        layer_time,_,starts=min(candidates,key=lambda x:(x[0],x[1]))
        for r in range(3):
            local=[[0.,*env.homes[r],'W']]
            if r in routes:
                if starts[r]>0: local.append([starts[r],*env.homes[r],'W'])
                local[-1][4]=routes[r][0][0][4]
                local.extend([[row[0]+starts[r],*row[1:]] for row in routes[r][0][1:]])
                trace.extend(dict(task=t['task'],robot_id=r+1,makespan_s=now+starts[r]+t['finish_local_s']) for t in routes[r][1])
            if local[-1][0]<layer_time: local.append([layer_time,*env.homes[r],'W'])
            archive[r][-1][4]=local[0][4]
            archive[r].extend([[row[0]+now,*row[1:]] for row in local[1:]])
        diagnostics.append(dict(z_mm=z,starts=starts,duration_s=layer_time,
                                serial_s=sum(duration.values()), forbidden_intervals={f'{a}-{b}':len(v) for (a,b),v in forbidden.items() if a<b}))
        now+=layer_time
        print(f'Proven segments z={z}: tasks={len(indices)}, makespan={now:.3f}',flush=True)
    env.rows,env.index,env.makespan,env.done=archive,len(env.tasks),now,True
    env.segment_diagnostics=diagnostics
    return trace
