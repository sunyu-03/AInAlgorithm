"""Train PPO to assign model-01 honeycomb tasks to three WAAM robots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from run_model_01 import (
    DEFAULT_JOB,
    DEFAULT_SCENARIO,
    DEFAULT_VALIDATOR_ROOT,
    build_trajectory,
    configure_validator_import,
    greedy_assign,
    honeycomb_paths,
    prepare_job,
)


HERE = Path(__file__).resolve().parent


def makespan(builders) -> float:
    return max(builder.last_time for builder in builders.values())


class HoneycombPPOEnv(gym.Env):
    """Discrete(3) environment matching AssignmentEnv's robot-selection action."""

    metadata = {"render_modes": []}

    def __init__(self, config, paths, layer_count: int, baseline_s: float):
        super().__init__()
        self.config = config
        self.paths = paths
        self.layer_count = layer_count
        self.baseline_s = baseline_s
        self.bases = np.asarray([config.robot(i).base_xyz_mm[:2] for i in (1, 2, 3)])
        self.reaches = np.asarray([config.robot(i).xy_reach_radius_mm for i in (1, 2, 3)])
        self.action_space = spaces.Discrete(3)
        # progress, current centroid XY, current length, three loads, three counts,
        # and distance from the current task to each robot base.
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(13,), dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        if self.index < len(self.paths):
            path = self.paths[self.index]
            center = np.asarray([path.centroid.x, path.centroid.y], dtype=float)
            length = float(path.length)
            distances = np.linalg.norm(self.bases - center, axis=1)
        else:
            center = np.zeros(2)
            length = 0.0
            distances = np.zeros(3)
        total_length = sum(float(path.length) for path in self.paths)
        return np.asarray(
            [
                self.index / len(self.paths),
                center[0] / 200.0,
                center[1] / 200.0,
                length / 100.0,
                *(self.loads / max(total_length, 1.0)),
                *(self.counts / len(self.paths)),
                *(distances / 2000.0),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.index = 0
        self.loads = np.zeros(3, dtype=float)
        self.counts = np.zeros(3, dtype=float)
        self.assignments = {1: [], 2: [], 3: []}
        return self._observation(), {}

    def step(self, action):
        robot = int(action)
        path = self.paths[self.index]
        center = np.asarray([path.centroid.x, path.centroid.y], dtype=float)
        distance = float(np.linalg.norm(center - self.bases[robot]))
        if distance > self.reaches[robot]:
            return self._observation(), -10.0, True, False, {"failure": "unreachable"}

        self.assignments[robot + 1].append(path)
        self.loads[robot] += float(path.length)
        self.counts[robot] += 1.0
        self.index += 1

        # Dense guidance: prefer nearby robots while avoiding extreme load imbalance.
        reward = -0.02 * distance / 1500.0
        reward -= 0.02 * float(np.std(self.loads)) / 420.0
        terminated = self.index == len(self.paths)
        info = {}
        if terminated:
            builders = build_trajectory(
                self.config, self.paths, self.layer_count, self.assignments
            )
            final_makespan = makespan(builders)
            # Main objective: lower makespan than the deterministic greedy baseline.
            reward += 10.0 * (self.baseline_s - final_makespan) / self.baseline_s
            reward += 2.0
            info = {"makespan_s": final_makespan}
        return self._observation(), float(reward), terminated, False, info


def assignment_indices(paths, assignments) -> list[int]:
    owner = {id(path): robot_id for robot_id, items in assignments.items() for path in items}
    return [owner[id(path)] for path in paths]


def assignments_from_policy(model: PPO, env: HoneycombPPOEnv):
    observation, _ = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.assignments, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator-root", type=Path, default=DEFAULT_VALIDATOR_ROOT)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validator_root = args.validator_root.expanduser().resolve()
    scenario = args.scenario.expanduser().resolve()
    job = args.job.expanduser().resolve()
    configure_validator_import(validator_root)
    from waam_validator import load_config
    from waam_validator.shape.target import load_target_mesh

    config = load_config(scenario / "config.yaml")
    mesh = load_target_mesh(scenario / "target.stl", config)
    layer_count = math.ceil(
        (float(mesh.bounds[1, 2]) - config.process.build_plane_z_mm)
        / config.process.layer_height_mm
    )
    paths = honeycomb_paths()
    greedy = greedy_assign(paths, config)
    greedy_s = makespan(build_trajectory(config, paths, layer_count, greedy))

    env = HoneycombPPOEnv(config, paths, layer_count, greedy_s)
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=300,
        batch_size=60,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model_path = HERE / "ppo_model_01"
    model.save(model_path)

    assignments, info = assignments_from_policy(model, env)
    ppo_builders = build_trajectory(config, paths, layer_count, assignments)
    ppo_s = makespan(ppo_builders)
    prepare_job(scenario, job, ppo_builders)

    record = {
        "algorithm": "PPO",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "observation_size": 13,
        "action_space": "Discrete(3): robot 1, 2, 3",
        "task_count": len(paths),
        "layer_count": layer_count,
        "greedy_baseline_makespan_s": greedy_s,
        "ppo_estimated_makespan_s": ppo_s,
        "improvement_s": greedy_s - ppo_s,
        "assignment_by_task": assignment_indices(paths, assignments),
        "tasks_per_robot": {
            str(robot_id): len(items) for robot_id, items in assignments.items()
        },
        "model_file": str(model_path.with_suffix(".zip")),
        "policy_terminal_info": info,
    }
    (HERE / "ppo_training_summary.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
