"""Surface-oriented perimeter and infill paths; independent of legacy tasks.

Contours use the inward bead-radius offset, including hole boundaries. The
nominal bead envelope is never clipped to the target to improve a score.
"""
from pathlib import Path
import numpy as np
from shapely import LineString, get_parts, union_all, line_interpolate_point, linestrings, STRtree
from .task_generator import DepositionTask, hatch_edges_parametric, merge_collinear
from .reach import split_by_reach


def envelope(edges, width, resolution):
    return union_all([LineString([a, b]).buffer(width / 2, quad_segs=resolution,
                       cap_style='round', join_style='round') for a, b in edges])


def surface_metrics(target, deposited, sample_mm=1.0):
    intersection = target.intersection(deposited).area
    def distances(a, b):
        samples = []
        segments = []
        for boundary in get_parts(b.boundary):
            coords = np.asarray(boundary.coords)
            segments.extend(linestrings(np.stack((coords[:-1], coords[1:]), axis=1)))
        tree = STRtree(segments)
        for line in get_parts(a.boundary):
            count = max(2, int(np.ceil(line.length / sample_mm)))
            points = line_interpolate_point(line, np.linspace(0, line.length, count, endpoint=False))
            # Exact nearest line-segment distance via a spatial index. The
            # query changes computational cost, not sampling or geometry.
            _, values = tree.query_nearest(points, all_matches=False, return_distance=True)
            samples.extend(values.tolist())
        return samples
    values = np.array(distances(target, deposited) + distances(deposited, target))
    return dict(coverage=intersection / target.area,
                overfill=deposited.difference(target).area / target.area,
                iou=intersection / target.union(deposited).area,
                boundary_p95_mm=float(np.percentile(values, 95)),
                boundary_max_sampled_mm=float(values.max()),
                boundary_sample_spacing_mm=sample_mm)


def perimeter_infill(poly, width, layer, resolution=8, tolerance_mm=0.05):
    if not 0 < tolerance_mm <= width / 16:
        raise ValueError('Contour tolerance must be positive and <= bead width / 16')
    inner = poly.buffer(-width / 2, quad_segs=resolution)
    if inner.is_empty:
        raise ValueError('Target thinner than fixed bead width; surface mode cannot resolve it')
    contours = []
    for part in get_parts(inner):
        for ring in (part.exterior, *part.interiors):
            coords = list(LineString(ring.coords).simplify(tolerance_mm).coords)
            contours.extend((np.asarray(a), np.asarray(b)) for a, b in zip(coords, coords[1:])
                            if np.linalg.norm(np.asarray(b)-a) > 1e-8)
    # Infill endpoints stay inside the same bead-radius offset. A perimeter
    # closes the gaps between round hatch endpoints at exterior and holes.
    infill = hatch_edges_parametric(poly, width, layer, 0.8, 0.5)
    return contours + infill


def generate_surface_tasks(stl, config, reach_margin_mm=1.0):
    from waam_validator.config.loader import load_config
    from waam_validator.shape.target import load_target_mesh, determine_evaluation_layers, slice_target_layers
    c = load_config(Path(config))
    mesh = load_target_mesh(Path(stl), c)
    polygons = slice_target_layers(mesh, determine_evaluation_layers(mesh, {}, c), c)
    records, checks = [], []
    for layer, poly in sorted(polygons.items()):
        if poly.is_empty:
            continue
        edges = perimeter_infill(poly, c.process.bead_width_mm, layer,
                                 c.shape_validation.polygon_buffer_resolution)
        edges = split_by_reach(merge_collinear(edges), c.robots, reach_margin_mm)
        quality = surface_metrics(poly, envelope(edges, c.process.bead_width_mm,
                                  c.shape_validation.polygon_buffer_resolution))
        sc = c.shape_validation
        if (quality['coverage'] < sc.minimum_overall_coverage or
            quality['overfill'] > sc.maximum_overall_overfill_ratio or
            quality['iou'] < max(sc.minimum_layer_iou, sc.minimum_overall_iou)):
            raise ValueError(f'Layer {layer}: surface paths fail shape precheck: {quality}')
        z = c.process.build_plane_z_mm + (layer + (0.5 if c.process.tcp_z_reference == 'center' else 1)) * c.process.layer_height_mm
        for a, b in edges:
            records.append(DepositionTask(len(records), layer, np.array([*a, z]), np.array([*b, z])))
        checks.append(dict(layer=layer, strategy='perimeter + infill(0.8w)',
                           contour_tolerance_mm=0.05, task_count=len(edges), **quality))
        print(f'Layer {layer}: surface tasks={len(edges)}, IoU={quality["iou"]:.6f}, boundary P95={quality["boundary_p95_mm"]:.4f}mm', flush=True)
    if not records:
        raise ValueError('No surface deposition tasks')
    return records, checks
