"""Generate and validate a greedy three-robot trajectory for public model 01.

Model 01 is the six-layer honeycomb target.  Each unique honeycomb edge is one
prepared deposition task.  The planner uses the same action meaning as the
week1 AssignmentEnv: choose robot 1, 2, or 3 for each task.  A deterministic
greedy policy balances assigned deposition length and distance from the robot
base.  Layers are processed bottom-up and robots are activated one at a time,
which gives a conservative collision-free baseline before later optimization.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import LineString


HERE = Path(__file__).resolve().parent
DEFAULT_VALIDATOR_ROOT = Path.home() / "Desktop" / "WAAM_Validator-main"
DEFAULT_SCENARIO = DEFAULT_VALIDATOR_ROOT / "tests" / "01"
DEFAULT_JOB = HERE.parent / "1" / "my_job"
DEFAULT_VALIDATION = HERE.parent / "1" / "validation_run"

Point2D = tuple[float, float]


@dataclass(slots=True)
class RobotRows:
    robot_id: int
    home: np.ndarray
    rows: list[tuple[float, float, float, float, str]]

    @classmethod
    def create(cls, robot_id: int, home: np.ndarray) -> "RobotRows":
        return cls(
            robot_id,
            home,
            [(0.0, float(home[0]), float(home[1]), float(home[2]), "W")],
        )

    @property
    def last_time(self) -> float:
        return self.rows[-1][0]

    @property
    def last_xyz(self) -> np.ndarray:
        return np.asarray(self.rows[-1][1:4], dtype=np.float64)

    def set_last_mode(self, mode: str) -> None:
        time_s, x_mm, y_mm, z_mm, _ = self.rows[-1]
        self.rows[-1] = (time_s, x_mm, y_mm, z_mm, mode)

    def append(self, time_s: float, xyz: np.ndarray, mode: str) -> None:
        if time_s <= self.last_time:
            raise ValueError(f"Robot {self.robot_id}: timestamps must increase")
        self.rows.append(
            (time_s, float(xyz[0]), float(xyz[1]), float(xyz[2]), mode)
        )


def configure_validator_import(validator_root: Path) -> None:
    source = validator_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Validator source directory not found: {source}")
    sys.path.insert(0, str(source))


def honeycomb_paths() -> list[LineString]:
    """Return the 30 unique centerlines used by public target 01."""
    radius = 42.0
    centers = [(0.0, 0.0)]
    for index in range(6):
        angle = index * math.pi / 3.0
        centers.append(
            (
                math.sqrt(3.0) * radius * math.cos(angle),
                math.sqrt(3.0) * radius * math.sin(angle),
            )
        )

    unique: dict[tuple[Point2D, Point2D], LineString] = {}
    for center_x, center_y in centers:
        vertices = [
            (
                center_x + radius * math.cos(math.pi / 6.0 + i * math.pi / 3.0),
                center_y + radius * math.sin(math.pi / 6.0 + i * math.pi / 3.0),
            )
            for i in range(6)
        ]
        for index in range(6):
            line = LineString([vertices[index], vertices[(index + 1) % 6]])
            left = tuple(round(float(value), 6) for value in line.coords[0])
            right = tuple(round(float(value), 6) for value in line.coords[-1])
            unique[tuple(sorted((left, right)))] = line
    paths = list(unique.values())
    if len(paths) != 30:
        raise RuntimeError(f"Expected 30 honeycomb edges, got {len(paths)}")
    return paths


def home_position(config, robot_id: int) -> np.ndarray:
    robot = config.robot(robot_id)
    if robot.home_xyz_mm is not None:
        return np.asarray(robot.home_xyz_mm, dtype=np.float64)
    raise ValueError(f"Robot {robot_id} has no configured home position")


def greedy_assign(paths: list[LineString], config) -> dict[int, list[LineString]]:
    """Discrete assignment action: select one of robot IDs 1, 2, or 3."""
    assignments = {1: [], 2: [], 3: []}
    loads = {1: 0.0, 2: 0.0, 3: 0.0}
    bases = {
        robot.id: np.asarray(robot.base_xyz_mm[:2], dtype=np.float64)
        for robot in config.robots
    }
    for path in sorted(paths, key=lambda item: (-item.length, item.centroid.x, item.centroid.y)):
        center = np.asarray([path.centroid.x, path.centroid.y], dtype=np.float64)
        reachable = [
            robot_id
            for robot_id in (1, 2, 3)
            if np.linalg.norm(center - bases[robot_id])
            <= config.robot(robot_id).xy_reach_radius_mm
        ]
        if not reachable:
            raise ValueError(f"No robot can reach path centered at {center.tolist()}")
        robot_id = min(
            reachable,
            key=lambda candidate: (
                loads[candidate] + float(np.linalg.norm(center - bases[candidate])),
                loads[candidate],
                candidate,
            ),
        )
        assignments[robot_id].append(path)
        loads[robot_id] += float(path.length)
    return assignments


def nearest_neighbor_order(paths: list[LineString], start_xy: np.ndarray) -> list[list[Point2D]]:
    remaining = [list(path.coords) for path in paths]
    ordered: list[list[Point2D]] = []
    current = start_xy.astype(np.float64)
    while remaining:
        best_index = 0
        reverse = False
        best_distance = float("inf")
        for index, coordinates in enumerate(remaining):
            start = np.asarray(coordinates[0], dtype=np.float64)
            end = np.asarray(coordinates[-1], dtype=np.float64)
            start_distance = float(np.linalg.norm(start - current))
            end_distance = float(np.linalg.norm(end - current))
            if min(start_distance, end_distance) < best_distance:
                best_index = index
                reverse = end_distance < start_distance
                best_distance = min(start_distance, end_distance)
        selected = remaining.pop(best_index)
        if reverse:
            selected.reverse()
        ordered.append([(float(point[0]), float(point[1])) for point in selected])
        current = np.asarray(selected[-1], dtype=np.float64)
    return ordered


def append_wait(builder: RobotRows, duration_s: float, next_mode: str) -> None:
    if duration_s > 0.0:
        builder.append(builder.last_time + duration_s, builder.last_xyz, next_mode)
    else:
        builder.set_last_mode(next_mode)


def append_travel(builder: RobotRows, target: np.ndarray, speed_mm_s: float, next_mode: str) -> None:
    distance = float(np.linalg.norm(target - builder.last_xyz))
    if distance > 1.0e-9:
        builder.append(builder.last_time + distance / speed_mm_s, target, next_mode)
    else:
        builder.set_last_mode(next_mode)


def append_layer_work(
    builder: RobotRows,
    paths: list[LineString],
    z_mm: float,
    start_s: float,
    config,
) -> float:
    """Run one robot while the other two remain parked at their homes."""
    if builder.last_time < start_s:
        builder.append(start_s, builder.home, "T")
    else:
        builder.set_last_mode("T")

    ordered = nearest_neighbor_order(paths, builder.home[:2])
    arc_on = config.process.arc_on_time_s or 0.0
    arc_off = config.process.arc_off_time_s or 0.0
    for coordinates in ordered:
        start = np.asarray([coordinates[0][0], coordinates[0][1], z_mm], dtype=np.float64)
        append_travel(builder, start, config.process.travel_speed_mm_s, "W")
        append_wait(builder, arc_on, "D")
        for x_mm, y_mm in coordinates[1:]:
            endpoint = np.asarray([x_mm, y_mm, z_mm], dtype=np.float64)
            append_travel(builder, endpoint, config.process.deposition_speed_mm_s, "D")
        builder.set_last_mode("W")
        append_wait(builder, arc_off, "T")
    append_travel(builder, builder.home, config.process.travel_speed_mm_s, "W")
    return builder.last_time


def build_trajectory(
    config,
    paths: list[LineString],
    layer_count: int,
    assignments: dict[int, list[LineString]] | None = None,
) -> dict[int, RobotRows]:
    builders = {
        robot_id: RobotRows.create(robot_id, home_position(config, robot_id))
        for robot_id in (1, 2, 3)
    }
    assignments = assignments if assignments is not None else greedy_assign(paths, config)
    global_time = 0.0
    for layer_index in range(layer_count):
        z_mm = (
            config.process.build_plane_z_mm
            + (layer_index + 0.5) * config.process.layer_height_mm
        )
        for robot_id in (1, 2, 3):
            if assignments[robot_id]:
                global_time = append_layer_work(
                    builders[robot_id], assignments[robot_id], z_mm, global_time, config
                )

    for builder in builders.values():
        if builder.last_time < global_time:
            builder.append(global_time, builder.home, "W")
        else:
            builder.set_last_mode("W")
    return builders


def write_trajectory(path: Path, builders: dict[int, RobotRows]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("robot_id", "time_s", "x_mm", "y_mm", "z_mm", "mode"))
        for robot_id in (1, 2, 3):
            for time_s, x_mm, y_mm, z_mm, mode in builders[robot_id].rows:
                writer.writerow(
                    (
                        robot_id,
                        f"{time_s:.9f}",
                        f"{x_mm:.6f}",
                        f"{y_mm:.6f}",
                        f"{z_mm:.6f}",
                        mode,
                    )
                )


def prepare_job(scenario: Path, job: Path, builders: dict[int, RobotRows]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "target.stl"):
        source = scenario / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, job / name)
    write_trajectory(job / "trajectory.csv", builders)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator-root", type=Path, default=DEFAULT_VALIDATOR_ROOT)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    args = parser.parse_args()

    validator_root = args.validator_root.expanduser().resolve()
    scenario = args.scenario.expanduser().resolve()
    job = args.job.expanduser().resolve()
    validation = args.validation.expanduser().resolve()
    configure_validator_import(validator_root)

    from waam_validator import load_config, run_validation
    from waam_validator.shape.target import load_target_mesh

    config = load_config(scenario / "config.yaml")
    mesh = load_target_mesh(scenario / "target.stl", config)
    layer_count = math.ceil(
        (float(mesh.bounds[1, 2]) - config.process.build_plane_z_mm)
        / config.process.layer_height_mm
    )
    paths = honeycomb_paths()
    builders = build_trajectory(config, paths, layer_count)
    prepare_job(scenario, job, builders)

    if validation.exists() and any(validation.iterdir()):
        raise FileExistsError(f"Validation directory must be empty: {validation}")
    result = run_validation(job, validation)
    print(f"status={result.status}")
    print(f"makespan_s={result.schedule.makespan_s:.6f}")
    print(f"coverage={result.shape.coverage:.9f}")
    print(f"iou={result.shape.iou:.9f}")
    print(f"collision_events={len(result.collision.events)}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
