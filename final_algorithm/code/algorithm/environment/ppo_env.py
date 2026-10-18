"""Versioned PPO contract: choose a checked robot for each serial home trip.

This deliberately has the same decoder during learning and export. It does not
optimize task order, continuous poses, or simultaneous robot motion.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from run_pipeline import AssignmentEnv, Task
from waam_validator import validate_trajectory_set, compute_reach_metrics
from waam_validator.collision.simulator import run_collision_analysis

CONTRACT = {
    'id': 'waam-serial-masked-v1',
    'observation_size': 16,
    'features': ['progress', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
                 'length', 'duration_robot1', 'duration_robot2', 'duration_robot3',
                 'mask_robot1', 'mask_robot2', 'mask_robot3', 'travel_speed', 'deposition_speed'],
    'normalization': 'xyz and length / max absolute scenario coordinate; durations / max trip duration; speeds / max process speed',
    'actions': 'Discrete(3): 0/1/2 -> robot 1/2/3; one task, home -> start -> deposit -> home',
    'reward': '-selected_trip_duration / max_trip_duration',
    'scheduler': 'serial, fixed task order, no overlap',
}

class SerialPPOEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, job, tasks: list[Task], reach_margin_mm=1.0):
        super().__init__()
        if not np.isfinite(reach_margin_mm) or reach_margin_mm < 0:
            raise ValueError('reach_margin_mm must be finite and nonnegative')
        self.backend = AssignmentEnv(job, tasks)
        self.tasks = self.backend.tasks
        self.margin = float(reach_margin_mm)
        self.trips = []
        self.durations = np.zeros((len(tasks), 3), dtype=float)
        self.masks = np.zeros((len(tasks), 3), dtype=bool)
        for index, task in enumerate(tasks):
            self.backend.index = index
            self.backend.rows = [[[0.0, *h, 'W']] for h in self.backend.homes]
            candidates = []
            for robot, cfg in enumerate(self.backend.robots):
                self.durations[index, robot] = self.backend._duration(task, robot)
                rows = None
                radius = cfg.xy_reach_radius_mm - self.margin
                if all(np.linalg.norm(np.asarray(p[:2])-self.backend.bases[robot,:2]) <= radius+1e-7 for p in (task.start, task.end)):
                    trial = self.backend._append_trip(robot, 0.0)
                    trajectory = self.backend._trajectory(trial)
                    if (not validate_trajectory_set(trajectory, self.backend.config).violations
                            and compute_reach_metrics(trajectory, self.backend.config).passed
                            and not run_collision_analysis(trajectory, self.backend.config).events):
                        rows = trial[robot]
                candidates.append(rows)
                self.masks[index, robot] = rows is not None
            if not self.masks[index].any():
                raise ValueError(f'No officially checked serial action for task {index}')
            self.trips.append(candidates)
        self.duration_scale = max(1.0, float(self.durations.max()))
        self.coord_scale = self.backend.coord_scale
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        self.makespan = 0.0
        self.done = False
        self.rows = [[[0.0, *h, 'W']] for h in self.backend.homes]
        self.trace = []
        return self._observation(), {}

    def action_masks(self):
        return self.masks[self.index].copy() if not self.done else np.zeros(3, dtype=bool)

    def _observation(self):
        if self.done:
            return np.zeros(16, dtype=np.float32)
        task = self.tasks[self.index]
        process = self.backend.config.process
        speed_scale = max(process.travel_speed_mm_s, process.deposition_speed_mm_s)
        return np.asarray([self.index/len(self.tasks),
                           *(np.asarray(task.start)/self.coord_scale),
                           *(np.asarray(task.end)/self.coord_scale),
                           np.linalg.norm(np.asarray(task.end)-task.start)/self.coord_scale,
                           *(self.durations[self.index]/self.duration_scale),
                           *self.action_masks().astype(float),
                           process.travel_speed_mm_s/speed_scale,
                           process.deposition_speed_mm_s/speed_scale], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended; call reset()')
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            # Reject without silently changing the action or state.
            raise ValueError('Action must be one of the officially checked masked actions')
        robot = int(action)
        trip = self.trips[self.index][robot]
        own = self.rows[robot]
        if self.makespan > own[-1][0]:
            own.append([self.makespan, *self.backend.homes[robot], 'W'])
        own[-1][4] = trip[0][4]
        own.extend([[r[0]+self.makespan, *r[1:]] for r in trip[1:]])
        duration = float(trip[-1][0])
        self.makespan += duration
        self.trace.append(dict(task=self.index, robot_id=robot+1, makespan_s=self.makespan))
        self.index += 1
        self.done = self.index == len(self.tasks)
        return self._observation(), -duration/self.duration_scale, self.done, False, {'makespan_s': self.makespan}

    def export(self, out):
        if not self.done:
            raise RuntimeError('Cannot export incomplete PPO schedule')
        self.backend.rows = self.rows
        self.backend.index = self.index
        self.backend.makespan = self.makespan
        return self.backend.export(out)

    def close(self):
        self.backend.close()
