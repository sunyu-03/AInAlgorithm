"""Home-ended spatial chunks; continuous hull proofs, immutable deposition paths.

Search includes the whole-layer schedule, so extra home trips cannot force a
regression. No velocity, bead, geometry or official validation setting changes.
"""
from itertools import combinations, permutations
import numpy as np
from shapely import MultiPoint
from .parallel_layers import nearest_route


def schedule_proven_chunks(env):
    archive = [[[0., *h, 'W']] for h in env.homes]
    now = 0.; trace = []; diagnostics = []
    collision = env.config.collision
    eps = max(collision.geometry_epsilon_mm, 1e-5)

    def hull(rows, r):
        points = np.asarray([row[1:3] for row in rows], np.float32).astype(float)
        return (MultiPoint(points).convex_hull,
                MultiPoint(np.vstack((points, env.bases[r, :2]))).convex_hull)

    def separated(a, b, ha, hb):
        return ((not collision.check_tcp_radius or ha[0].distance(hb[0]) >
                 env.robots[a].tcp_radius_mm + env.robots[b].tcp_radius_mm + eps)
                and (not collision.check_arm_envelope or ha[1].distance(hb[1]) >
                     env.robots[a].arm_envelope_radius_mm + env.robots[b].arm_envelope_radius_mm
                     + collision.arm_clearance_mm + eps))

    parked = [hull([[0., *env.homes[r], 'W']], r) for r in range(3)]
    for a, b in combinations(range(3), 2):
        if not separated(a, b, parked[a], parked[b]):
            raise RuntimeError('Cannot prove parked robot separation')
    layers = {}
    for i, task in enumerate(env.tasks):
        layers.setdefault(task.start[2], []).append(i)
    for z, indices in sorted(layers.items()):
        groups = [[], [], []]
        for i in indices:
            t = env.tasks[i]
            eligible = [r for r in range(3) if all(np.linalg.norm(np.asarray(p[:2])-env.bases[r,:2]) <=
                       env.robots[r].xy_reach_radius_mm-env.reach_margin_mm+1e-7 for p in (t.start,t.end))]
            if not eligible: raise ValueError(f'Unreachable task {i}')
            mid = (np.asarray(t.start)+t.end)/2
            r = min(eligible, key=lambda r: (np.linalg.norm(mid[:2]-env.bases[r,:2]), r))
            groups[r].append(i)
        whole = [nearest_route(env, g, r) if g else None for r, g in enumerate(groups)]
        ordered = [[t['task'] for t in route[1]] if route else [] for route in whole]
        best = None; search = []
        for size in (0, 32, 64, 128, 256):
            jobs = []; queues = [[], [], []]
            for r, order in enumerate(ordered):
                parts = [order] if size == 0 else [order[k:k+size] for k in range(0,len(order),size)]
                for part in parts:
                    if not part: continue
                    rows, detail = whole[r] if size == 0 else nearest_route(env, part, r)
                    shape = hull(rows, r)
                    for other in range(3):
                        if other != r and not separated(r, other, shape, parked[other]):
                            raise RuntimeError(f'Cannot prove chunk/home separation at z={z}')
                    queues[r].append(len(jobs))
                    jobs.append(dict(robot=r, rows=rows, trace=detail, hull=shape, duration=rows[-1][0]))
            conflicts = {}
            def conflict(a, b):
                key = tuple(sorted((a,b)))
                if key not in conflicts:
                    ja, jb = jobs[a], jobs[b]
                    conflicts[key] = (ja['robot'] == jb['robot'] or not separated(
                        ja['robot'], jb['robot'], ja['hull'], jb['hull']))
                return conflicts[key]
            # Six cyclic dispatch orders and an earliest-finish dispatcher.
            for priority in (*permutations(range(3)), None):
                cursor = [0,0,0]; ends = [0.,0.,0.]; reservations = []; turn = 0
                while sum(cursor) < len(jobs):
                    options = []
                    for r in range(3):
                        if cursor[r] >= len(queues[r]): continue
                        j = queues[r][cursor[r]]; start = ends[r]; duration = jobs[j]['duration']
                        for other, left, right in sorted(reservations, key=lambda v:v[1]):
                            if start < right and start+duration > left and conflict(j, other):
                                start = right
                        options.append((start+duration, start, r, j))
                    if priority is None:
                        finish, start, r, j = min(options)
                    else:
                        available = {o[2]:o for o in options}
                        while priority[turn%3] not in available: turn += 1
                        finish, start, r, j = available[priority[turn%3]]; turn += 1
                    reservations.append((j,start,finish)); ends[r]=finish; cursor[r]+=1
                duration = max(ends)
                if best is None or duration < best[0]-1e-9:
                    best = (duration, size, jobs, reservations)
            search.append(dict(chunk_size=size, best_s=best[0]))
        duration, size, jobs, reservations = best
        local = [[[0.,*h,'W']] for h in env.homes]
        for j, start, finish in sorted(reservations, key=lambda v:v[1]):
            job = jobs[j]; r = job['robot']; rows = local[r]
            if start > rows[-1][0]: rows.append([start,*env.homes[r],'W'])
            rows[-1][4] = job['rows'][0][4]
            rows.extend([[row[0]+start,*row[1:]] for row in job['rows'][1:]])
            trace.extend(dict(task=t['task'], robot_id=r+1, makespan_s=now+start+t['finish_local_s'])
                         for t in job['trace'])
        for r in range(3):
            if local[r][-1][0] < duration: local[r].append([duration,*env.homes[r],'W'])
            archive[r][-1][4] = local[r][0][4]
            archive[r].extend([[row[0]+now,*row[1:]] for row in local[r][1:]])
        diagnostics.append(dict(z_mm=z, chunk_size=size, duration_s=duration, search=search))
        now += duration
        print(f'Proven chunks z={z}: size={size}, tasks={len(indices)}, makespan={now:.3f}', flush=True)
    env.rows, env.index, env.makespan, env.done = archive, len(env.tasks), now, True
    env.chunk_diagnostics = diagnostics
    return trace
