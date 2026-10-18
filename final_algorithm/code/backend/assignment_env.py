"""Small fixed-order WAAM scheduling environment, using the official validator.

Each action assigns ONE prepared deposition segment to a robot. Every job is
a home -> start -> deposit -> home trip. Trips may overlap only when the
official sampled collision check accepts them. This is a baseline, not a slicer.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# The team's env module selects the bundled backend and refuses mixed versions.
from environment import env as team_checks

VALIDATOR_ROOT = Path(team_checks.__file__).resolve().parent
from waam_validator import (
    load_config,
    validate_trajectory_set,
    compute_reach_metrics,
    run_validation as run_validation,
)
from waam_validator.collision.simulator import run_collision_analysis
from waam_validator.models import RobotTrajectory, TrajectorySet


@dataclass(frozen=True)
class Task:
    start: tuple[float, float, float]
    end: tuple[float, float, float]


class AssignmentEnv(gym.Env):
    """Discrete(3): 0/1/2 maps to robot IDs 1/2/3.

    Observation contains all prepared tasks and accepted allocations/times, plus
    static robot/process data. It does not hide the accepted schedule history.
    Tasks must be ordered by increasing layer height. Their coverage is checked
    by the full Validator on export, NOT claimed from step() success alone.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, scenario: str | Path, tasks: list[Task], *, randomize_direction=False
    ):
        super().__init__()
        self.scenario = Path(scenario).resolve()
        self.config = load_config(self.scenario / "config.yaml")
        if not (self.scenario / "target.stl").is_file():
            raise ValueError("scenario must contain target.stl")
        if not tasks:
            raise ValueError("At least one prepared deposition task is required")
        self.original_tasks = list(tasks)
        self.tasks = list(tasks)
        self.randomize_direction = randomize_direction
        self.robots = sorted(self.config.robots, key=lambda r: r.id)
        self.homes = np.array(
            [
                r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm
                for r in self.robots
            ],
            dtype=float,
        )
        self.bases = np.array([r.base_xyz_mm for r in self.robots], dtype=float)
        self.coord_scale = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        np.concatenate(
                            [
                                self.homes,
                                self.bases,
                                np.array([(t.start, t.end) for t in tasks]).reshape(
                                    -1, 3
                                ),
                            ]
                        )
                    )
                )
            ),
        )
        self._validate_tasks()
        # Sum of worst round-trip durations bounds our serial fallback schedule.
        self.time_scale = max(
            1.0, sum(max(self._duration(task, r) for r in range(3)) for task in tasks)
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 11 * len(tasks) + 3 * 9 + 4,), dtype=np.float32
        )
        self.reset()
        initial = self._trajectory(self.rows)
        if (
            not compute_reach_metrics(initial, self.config).passed
            or run_collision_analysis(initial, self.config).events
        ):
            raise ValueError(
                "Initial parked robot positions violate reach/collision rules"
            )

    def _validate_tasks(self):
        last_z = -np.inf
        p = self.config.process
        for task in self.tasks:
            pts = np.array([task.start, task.end], dtype=float)
            if pts.shape != (2, 3) or not np.isfinite(pts).all():
                raise ValueError("Task endpoints must be finite XYZ triples")
            if abs(pts[0, 2] - pts[1, 2]) > 1e-8 or pts[0, 2] < last_z:
                raise ValueError("Tasks must be horizontal and sorted by layer Z")
            last_z = pts[0, 2]
            duration = np.linalg.norm(pts[1] - pts[0]) / p.deposition_speed_mm_s
            if duration <= 0:
                raise ValueError("Zero-length deposition task")
            # Reuse official layer/workspace validation; reach is action-specific.
            rows = [
                [[0.0, *self.homes[r], "W"], [1.0, *self.homes[r], "W"]]
                for r in range(3)
            ]
            rows[0] = [[0.0, *pts[0], "D"], [duration, *pts[1], "W"]]
            messages = validate_trajectory_set(self._trajectory(rows), self.config)
            # Robot 1 is only a placeholder for checking deposition geometry.
            # Actual reach depends on the robot chosen later in candidate().
            violations = [issue for issue in messages.violations
                          if issue.code != "ROBOT_XY_REACH_VIOLATION"]
            if violations:
                raise ValueError("Invalid task: " + ", ".join(
                    issue.code for issue in violations
                ))

    def _duration(self, task, robot):
        p = self.config.process
        return float(
            (
                np.linalg.norm(np.array(task.start) - self.homes[robot])
                + np.linalg.norm(np.array(task.end) - self.homes[robot])
            )
            / p.travel_speed_mm_s
            + np.linalg.norm(np.array(task.end) - task.start) / p.deposition_speed_mm_s
        )

    @staticmethod
    def _trajectory(rows):
        trajectories = []
        for robot, source in enumerate(rows, start=1):
            source = [list(row) for row in source]
            if len(source) == 1:
                source.append([max(1e-6, source[0][0] + 1e-6), *source[0][1:4], "W"])
            trajectories.append(
                RobotTrajectory(
                    robot,
                    np.array([r[0] for r in source], dtype=np.float64),
                    np.array([r[1:4] for r in source], dtype=np.float32),
                    np.array(
                        [{"T": 0, "D": 1, "W": 2}[r[4]] for r in source], dtype=np.uint8
                    ),
                )
            )
        return TrajectorySet(
            tuple(trajectories), sum(len(t.time_s) for t in trajectories)
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tasks = [
            Task(t.end, t.start)
            if self.randomize_direction and self.np_random.random() < 0.5
            else t
            for t in self.original_tasks
        ]
        self.rows = [[[0.0, *home, "W"]] for home in self.homes]
        self.allocations = np.zeros((len(self.tasks), 4), dtype=float)
        self.index = 0
        self.makespan = 0.0
        self.done = False
        return self._observation(), self._info()

    def _observation(self):
        features = [self.index / len(self.tasks)]
        for k, task in enumerate(self.tasks):
            features.extend(np.array([*task.start, *task.end]) / self.coord_scale)
            features.append(float(k < self.index))
            features.extend(self.allocations[k])
        for r, robot in enumerate(self.robots):
            features.extend(self.bases[r] / self.coord_scale)
            features.extend(self.homes[r] / self.coord_scale)
            features.extend(
                np.array(
                    [
                        robot.xy_reach_radius_mm,
                        robot.arm_envelope_radius_mm,
                        robot.tcp_radius_mm,
                    ]
                )
                / self.coord_scale
            )
        features.extend(
            [
                self.config.process.deposition_speed_mm_s / self.coord_scale,
                self.config.process.travel_speed_mm_s / self.coord_scale,
                self.config.collision.arm_clearance_mm / self.coord_scale,
                self.makespan / self.time_scale,
            ]
        )
        return np.array(features, dtype=np.float32)

    def _info(self, **extra):
        return dict(
            makespan_s=self.makespan,
            completed_tasks=self.index,
            total_tasks=len(self.tasks),
            schedule_complete=self.index == len(self.tasks),
            **extra,
        )

    def _append_trip(self, robot, departure):
        rows = [[list(row) for row in source] for source in self.rows]
        own = rows[robot]
        if departure > own[-1][0] + 1e-9:
            own.append([departure, *self.homes[robot], "W"])
        task = self.tasks[self.index]
        for point, mode, speed in [
            (task.start, "T", self.config.process.travel_speed_mm_s),
            (task.end, "D", self.config.process.deposition_speed_mm_s),
            (self.homes[robot], "T", self.config.process.travel_speed_mm_s),
        ]:
            distance = float(np.linalg.norm(np.array(point) - np.array(own[-1][1:4])))
            if distance < 1e-9:
                continue
            own[-1][4] = mode
            own.append([own[-1][0] + distance / speed, *point, "W"])
        return rows

    def candidate(self, robot):
        """Try event times, ending with a conservative serial fallback.

        Every candidate is checked against ALL trajectories, including waiting
        and parked robots. No promise of the earliest continuous-time start.
        """
        if self.done or not self.action_space.contains(robot):
            return None, "invalid_action", 0.0
        task = self.tasks[self.index]
        r = self.robots[robot]
        for point in (task.start, task.end):
            if (
                np.linalg.norm(np.array(point[:2]) - self.bases[robot, :2])
                > r.xy_reach_radius_mm + self.config.collision.geometry_epsilon_mm
            ):
                return None, "unreachable", 0.0
        # All jobs in earlier layers must finish before the new layer starts.
        barrier = max(
            [0.0]
            + [
                self.allocations[k, 3] * self.time_scale
                for k in range(self.index)
                if self.tasks[k].start[2] < task.start[2] - 1e-8
            ]
        )
        earliest = max(self.rows[robot][-1][0], barrier)
        starts = sorted(
            {
                earliest,
                max(earliest, self.makespan),
                *(
                    row[0]
                    for source in self.rows
                    for row in source
                    if row[0] >= earliest
                ),
            }
        )
        for departure in starts:
            rows = self._append_trip(robot, departure)
            trajectory = self._trajectory(rows)
            messages = validate_trajectory_set(trajectory, self.config)
            if (
                messages.violations
                or not compute_reach_metrics(trajectory, self.config).passed
            ):
                continue
            if not run_collision_analysis(trajectory, self.config).events:
                return rows, "ok", departure
        return None, "collision_or_constraint", 0.0

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1 or 2")
        robot = int(action)
        rows, reason, departure = self.candidate(robot)
        if rows is None:
            self.done = True
            return (
                self._observation(),
                -2.0,
                True,
                False,
                self._info(failure_reason=reason),
            )
        previous = self.makespan
        self.rows = rows
        finish = rows[robot][-1][0]
        self.makespan = max(previous, finish)
        self.allocations[self.index] = [
            (robot + 1) / 3.0,
            departure / self.time_scale,
            self._duration(self.tasks[self.index], robot) / self.time_scale,
            finish / self.time_scale,
        ]
        self.index += 1
        self.done = self.index == len(self.tasks)
        reward = -(self.makespan - previous) / self.time_scale
        if self.done:
            reward += 1.0
        return self._observation(), float(reward), self.done, False, self._info()

    def export(self, output: str | Path):
        if self.index != len(self.tasks):
            raise RuntimeError("Cannot export an incomplete schedule")
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a new/empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "target.stl"):
            shutil.copy2(self.scenario / name, output / name)
        with (output / "trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["robot_id", "time_s", "x_mm", "y_mm", "z_mm", "mode"])
            for robot, source in enumerate(self.rows, start=1):
                rows = [list(row) for row in source]
                if len(rows) == 1:
                    rows.append([self.makespan, *rows[0][1:4], "W"])
                writer.writerows([[robot, *row] for row in rows])
        return output


def load_tasks(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Task(tuple(t["start"]), tuple(t["end"])) for t in records]


def demo_tasks():
    # Exactly partition the supplied sample's -40..40 mm deposition centerline.
    return [
        Task((x, 0.0, 2.0), (x + 20.0, 0.0, 2.0)) for x in (-40.0, -20.0, 0.0, 20.0)
    ]


def greedy_action(env):
    scores = []
    for robot in range(3):
        rows, _, _ = env.candidate(robot)
        scores.append(
            max(source[-1][0] for source in rows) if rows is not None else np.inf
        )
    return int(np.argmin(scores))
