"""Explicit discrete task-allocation adapter; existing PPO weights are incompatible."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.deps'))
sys.path.insert(0, str(ROOT.parent/'TEAM_PROJECT/week1'))
from assignment_env import AssignmentEnv, Task
import numpy as np

class WAAMBaselineEnv(AssignmentEnv):
    def __init__(self, scenario):
        job = Path(scenario['job_dir'])
        if Path(scenario['stl_path']) != job/'target.stl' or Path(scenario['config_path']) != job/'config.yaml':
            raise ValueError('Scenario requires target.stl and config.yaml in the same job directory')
        tasks = [Task(tuple(t['start_xyz_mm']), tuple(t['end_xyz_mm'])) for t in scenario['tasks']]
        self.reach_margin_mm = float(scenario.get('reach_margin_mm', 1.0))
        if not np.isfinite(self.reach_margin_mm) or self.reach_margin_mm < 0:
            raise ValueError('reach margin must be finite and non-negative')
        super().__init__(job, tasks)
        self.ppo_ready = False

    def reset(self, **kwargs):
        self._candidate_cache = {}
        return super().reset(**kwargs)

    def candidate(self, robot):
        if self.done or not self.action_space.contains(robot):
            return super().candidate(robot)
        task = self.tasks[self.index]
        if any(np.linalg.norm(np.asarray(p[:2])-self.bases[robot,:2]) > self.robots[robot].xy_reach_radius_mm-self.reach_margin_mm+1e-7 for p in (task.start,task.end)):
            return None, 'reach_margin', 0.0
        key = (self.index, int(robot))
        if key not in self._candidate_cache:
            self._candidate_cache[key] = super().candidate(robot)
        return self._candidate_cache[key]

    def step(self, action):
        result = super().step(action)
        self._candidate_cache.clear()
        return result
