"""Select shorter infill without losing measured surface quality.

Unrepresentable thin sections retain shape-validated legacy paths explicitly.
No process settings or official validator thresholds are changed.
"""
from pathlib import Path
import numpy as np
from shapely import LineString
from .surface_generator import perimeter_infill, envelope, surface_metrics
from .task_generator import DepositionTask, hatch_edges_parametric, choose_edges, merge_collinear
from .reach import split_by_reach


def choose_efficient(poly, width, layer, sc):
    resolution = sc.polygon_buffer_resolution
    def assess(edges):
        return surface_metrics(poly, envelope(edges, width, resolution))
    def feasible(q):
        return (q['coverage'] >= sc.minimum_overall_coverage and
                q['overfill'] <= sc.maximum_overall_overfill_ratio and
                q['iou'] >= max(sc.minimum_layer_iou, sc.minimum_overall_iou))
    try:
        reference = perimeter_infill(poly, width, layer, resolution)
    except ValueError:
        reference = []
    quality = assess(reference) if reference else None
    if quality is None or not feasible(quality):
        legacy, strategy, _ = choose_edges(poly, width, layer, sc.model_dump())
        q = assess(legacy)
        if not feasible(q):
            raise ValueError('Thin-section legacy fallback fails exact bead precheck')
        return legacy, 'THIN_FEATURE_LEGACY_FALLBACK: '+strategy, q
    # The old profile is a fixed feasible fallback. Try less overlapping
    # hatches and shorter endpoints, but reject any new internal gaps that
    # would pass volume IoU while worsening the actual boundary distances.
    original_infill = hatch_edges_parametric(poly, width, layer, .8, .5)
    contours = reference[:-len(original_infill)] if original_infill else reference
    pool = [(reference, 'surface(0.8w,0.5w)', quality)]
    for spacing, inset in ((.98,.5), (.98,.65), (.98,.75), (.95,.75)):
        trial = contours + hatch_edges_parametric(poly, width, layer, spacing, inset)
        dep = envelope(trial, width, resolution)
        # Exact area prerequisite cheaply rejects interior-gap candidates;
        # it does not change the admissibility test or its thresholds.
        iou = poly.intersection(dep).area / poly.union(dep).area
        if iou < quality['iou']-1e-5:
            continue
        q = surface_metrics(poly, dep)
        if (feasible(q) and q['iou'] >= quality['iou']-1e-5 and
            q['boundary_p95_mm'] <= quality['boundary_p95_mm']+.02 and
            q['boundary_max_sampled_mm'] <= quality['boundary_max_sampled_mm']+.10):
            pool.append((trial, f'surface({spacing}w,{inset}w)', q))
    return min(pool, key=lambda x: sum(LineString(e).length for e in x[0]))


def generate_efficient_surface_tasks(stl, config, reach_margin_mm=1.0):
    from waam_validator.config.loader import load_config
    from waam_validator.shape.target import load_target_mesh, determine_evaluation_layers, slice_target_layers
    c = load_config(Path(config)); mesh = load_target_mesh(Path(stl), c)
    polygons = slice_target_layers(mesh, determine_evaluation_layers(mesh, {}, c), c)
    records, checks, cache = [], [], {}
    for layer, poly in sorted(polygons.items()):
        if poly.is_empty: continue
        # Exact geometry key only: differing sections never inherit a path.
        key = (layer % 2, poly.wkb)
        if key in cache:
            edges, strategy, quality = cache[key]
        else:
            edges, strategy, quality = choose_efficient(poly, c.process.bead_width_mm, layer, c.shape_validation)
            edges = split_by_reach(merge_collinear(edges), c.robots, reach_margin_mm)
            cache[key] = edges, strategy, quality
        z = c.process.build_plane_z_mm + (layer+(0.5 if c.process.tcp_z_reference == 'center' else 1))*c.process.layer_height_mm
        for a,b in edges:
            records.append(DepositionTask(len(records),layer,np.array([*a,z]),np.array([*b,z])))
        checks.append(dict(layer=layer,strategy=strategy,task_count=len(edges),**quality))
        print(f'Layer {layer}: {strategy}, tasks={len(edges)}, IoU={quality["iou"]:.6f}',flush=True)
    if not records: raise ValueError('No tasks generated')
    return records, checks
