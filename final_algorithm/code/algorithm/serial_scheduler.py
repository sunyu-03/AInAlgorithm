"""Conservative serial scheduling with official checks on each independent trip."""
import numpy as np
from waam_validator import validate_trajectory_set, compute_reach_metrics
from waam_validator.collision.simulator import run_collision_analysis

def schedule_serial(env):
    # Every accepted trip starts and finishes at home. Thus all other robots are
    # parked throughout the local window; past motion cannot overlap this trip.
    all_rows = [[[0.0, *home, 'W']] for home in env.homes]
    now = 0.0
    trace = []
    for index, task in enumerate(env.tasks):
        env.index = index
        env.rows = [[[0.0, *home, 'W']] for home in env.homes]
        choices = []
        reasons = []
        for robot, config in enumerate(env.robots):
            if any(np.linalg.norm(np.asarray(point[:2])-env.bases[robot,:2]) > config.xy_reach_radius_mm-getattr(env,'reach_margin_mm',0.0)+1e-7 for point in (task.start, task.end)):
                reasons.append('unreachable')
                continue
            rows = env._append_trip(robot, 0.0)
            trajectory = env._trajectory(rows)
            if validate_trajectory_set(trajectory, env.config).violations or not compute_reach_metrics(trajectory, env.config).passed:
                reasons.append('trajectory_or_reach')
                continue
            if run_collision_analysis(trajectory, env.config).events:
                reasons.append('collision')
                continue
            choices.append((rows[robot][-1][0], robot, rows))
        if not choices:
            raise RuntimeError(f'No safe serial trip for task {index}: {reasons}')
        duration, robot, rows = min(choices, key=lambda item:(item[0], item[1]))
        own = all_rows[robot]
        if now > own[-1][0]:
            own.append([now, *env.homes[robot], 'W'])
        own[-1][4] = rows[robot][0][4]
        own.extend([[row[0]+now, *row[1:]] for row in rows[robot][1:]])
        now += duration
        trace.append(dict(task=index, robot_id=robot+1, makespan_s=now))
        if (index+1)%50 == 0 or index+1 == len(env.tasks):
            print(f'Serial task {index+1}/{len(env.tasks)}, makespan={now:.3f}', flush=True)
    env.rows = all_rows
    env.index = len(env.tasks)
    env.makespan = now
    env.done = True
    return trace

def schedule_layers(env):
    """Process one robot at a time per layer, retaining its position between beads.

    Each local candidate includes a checked return home. Only the work prefix is
    committed until the group ends. All other robots stay parked, so accepted
    windows compose without cross-window motion overlap.
    """
    rows = [[[0.0, *home, 'W']] for home in env.homes]
    now = 0.0
    trace = []
    layers = sorted(set(task.start[2] for task in env.tasks))
    for z in layers:
        pending = [i for i, task in enumerate(env.tasks) if task.start[2] == z]
        indices = np.asarray(pending, dtype=int)
        starts = np.asarray([env.tasks[i].start for i in indices],float)
        ends = np.asarray([env.tasks[i].end for i in indices],float)
        done = np.zeros(len(indices),dtype=bool)
        for robot in range(3):
            current = np.asarray(env.homes[robot], float)
            radius = env.robots[robot].xy_reach_radius_mm-getattr(env,'reach_margin_mm',0.0)+1e-7
            eligible = ((np.linalg.norm(starts[:,:2]-env.bases[robot,:2],axis=1) <= radius) &
                        (np.linalg.norm(ends[:,:2]-env.bases[robot,:2],axis=1) <= radius))
            while pending:
                positions = np.flatnonzero(eligible & ~done)
                if not len(positions):
                    break
                route_starts = np.concatenate((starts[positions],ends[positions]))
                route_ends = np.concatenate((ends[positions],starts[positions]))
                route_indices = np.tile(indices[positions],2)
                route_positions = np.tile(positions,2)
                distances = np.linalg.norm(route_starts-current,axis=1)
                order = np.lexsort((route_starts[:,2],route_starts[:,1],route_starts[:,0],route_indices,distances))
                accepted = None
                for choice in order:
                    index = int(route_indices[choice])
                    start, end = route_starts[choice], route_ends[choice]
                    work = [[0.0,*current,'W']]
                    for point, mode, speed in ((start,'T',env.config.process.travel_speed_mm_s),
                                               (end,'D',env.config.process.deposition_speed_mm_s)):
                        distance = float(np.linalg.norm(np.asarray(point)-np.asarray(work[-1][1:4])))
                        if distance > 1e-9:
                            work[-1][4] = mode
                            work.append([work[-1][0]+distance/speed,*point,'W'])
                    checked = [r.copy() for r in work]
                    distance = float(np.linalg.norm(np.asarray(end)-env.homes[robot]))
                    if distance > 1e-9:
                        checked[-1][4] = 'T'
                        checked.append([checked[-1][0]+distance/env.config.process.travel_speed_mm_s,*env.homes[robot],'W'])
                    local = [[[0.0,*home,'W'],[checked[-1][0],*home,'W']] for home in env.homes]
                    local[robot] = checked
                    trajectory = env._trajectory(local)
                    if (validate_trajectory_set(trajectory,env.config).violations or
                        not compute_reach_metrics(trajectory,env.config).passed or
                        run_collision_analysis(trajectory,env.config).events):
                        continue
                    accepted = index, end, work, int(route_positions[choice])
                    break
                if accepted is None:
                    break
                index, end, work, position = accepted
                done[position] = True
                own = rows[robot]
                if now > own[-1][0]:
                    own.append([now,*current,'W'])
                own[-1][4] = work[0][4]
                own.extend([[r[0]+now,*r[1:]] for r in work[1:]])
                now += work[-1][0]
                current = np.asarray(end,float)
                pending.remove(index)
                trace.append(dict(task=index,robot_id=robot+1,makespan_s=now))
            distance = float(np.linalg.norm(current-env.homes[robot]))
            if distance > 1e-9:
                rows[robot][-1][4] = 'T'
                now += distance/env.config.process.travel_speed_mm_s
                rows[robot].append([now,*env.homes[robot],'W'])
        if pending:
            raise RuntimeError(f'No safe layer route for {len(pending)} tasks at z={z}: {pending[:10]}')
        print(f'Layer route z={z}: complete={len(trace)}/{len(env.tasks)}, makespan={now:.3f}',flush=True)
    env.rows, env.index, env.makespan, env.done = rows, len(env.tasks), now, True
    return trace
