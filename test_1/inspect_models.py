"""Inspect the six professor-supplied WAAM models with the official loader."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from shapely import get_parts


HERE = Path(__file__).resolve().parent
DEFAULT_WEEK1 = HERE.parent / "TEAM_PROJECT" / "week1"
DEFAULT_MODELS = Path.home() / "Desktop" / "WAAM_Validator-main" / "tests"


def add_validator_to_path(week1: Path) -> Path:
    validator_src = week1 / "environment" / "src"
    if not validator_src.is_dir():
        raise FileNotFoundError(f"Validator source not found: {validator_src}")
    sys.path.insert(0, str(validator_src))
    return validator_src


def inspect_model(model_dir: Path) -> dict[str, object]:
    from waam_validator import load_config
    from waam_validator.shape.target import load_target_mesh, slice_target_layers

    config_path = model_dir / "config.yaml"
    target_path = model_dir / "target.stl"
    missing = [path.name for path in (config_path, target_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{model_dir.name}: missing {', '.join(missing)}")

    started = time.perf_counter()
    config = load_config(config_path)
    mesh = load_target_mesh(target_path, config)
    target_height = float(mesh.bounds[1, 2] - config.process.build_plane_z_mm)
    layer_count = math.ceil(target_height / config.process.layer_height_mm)
    layers = slice_target_layers(mesh, range(layer_count), config)

    areas: list[float] = []
    component_counts: list[int] = []
    hole_counts: list[int] = []
    for geometry in layers.values():
        if geometry.is_empty:
            continue
        components = list(get_parts(geometry))
        areas.append(float(geometry.area))
        component_counts.append(len(components))
        hole_counts.append(sum(len(getattr(part, "interiors", ())) for part in components))

    return {
        "model": model_dir.name,
        "files": {
            "config_yaml": str(config_path),
            "target_stl": str(target_path),
            "target_bytes": target_path.stat().st_size,
        },
        "mesh": {
            "triangles": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "watertight": bool(mesh.is_watertight),
            "body_count": int(mesh.metadata.get("waam_body_count", -1)),
            "bounds_mm": [[float(value) for value in row] for row in mesh.bounds],
            "height_mm": target_height,
        },
        "layers": {
            "count": layer_count,
            "nonempty_count": len(areas),
            "minimum_area_mm2": min(areas, default=0.0),
            "maximum_area_mm2": max(areas, default=0.0),
            "maximum_component_count": max(component_counts, default=0),
            "maximum_hole_count": max(hole_counts, default=0),
        },
        "inspection_seconds": time.perf_counter() - started,
    }


def inspect_all(models_root: Path) -> list[dict[str, object]]:
    expected = [f"{number:02d}" for number in range(1, 7)]
    actual = sorted(path.name for path in models_root.iterdir() if path.is_dir())
    missing = [name for name in expected if name not in actual]
    if missing:
        raise FileNotFoundError(f"Missing model directories: {', '.join(missing)}")
    return [inspect_model(models_root / name) for name in expected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--week1", type=Path, default=DEFAULT_WEEK1)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "model_inventory.json")
    args = parser.parse_args()

    models_root = args.models.expanduser().resolve()
    week1 = args.week1.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not models_root.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_root}")

    add_validator_to_path(week1)
    records = inspect_all(models_root)
    payload = {
        "models_root": str(models_root),
        "week1_root": str(week1),
        "model_count": len(records),
        "models": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        mesh = record["mesh"]
        layers = record["layers"]
        print(
            f"{record['model']}: triangles={mesh['triangles']}, "
            f"layers={layers['count']}, components<= {layers['maximum_component_count']}, "
            f"holes<= {layers['maximum_hole_count']}"
        )
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

