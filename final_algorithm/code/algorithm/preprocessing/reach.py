"""Exact straight-segment partitions within the union of circular XY reaches."""
import numpy as np

def split_by_reach(edges, robots, margin_mm=1.0):
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError('reach margin must be finite and non-negative')
    circles = [(np.asarray(r.base_xyz_mm[:2], float), r.xy_reach_radius_mm-margin_mm) for r in robots]
    if any(radius <= 0 for _, radius in circles):
        raise ValueError('reach margin must be smaller than every robot reach')
    result = []
    for edge_index, (a, b) in enumerate(edges):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b-a
        length2 = float(d@d)
        if length2 <= 1e-18:
            raise ValueError(f'Zero-length edge {edge_index}')
        cuts = [0.0, 1.0]
        intervals = []
        for center, radius in circles:
            q = a-center
            linear = float(q@d)
            discriminant = linear*linear-length2*(float(q@q)-radius*radius)
            if discriminant < 0:
                intervals.append(None)
                continue
            root = np.sqrt(max(0.0, discriminant))
            lo, hi = max(0.0, (-linear-root)/length2), min(1.0, (-linear+root)/length2)
            intervals.append((lo, hi) if lo <= hi else None)
            if lo <= hi:
                cuts.extend([lo, hi])
        if any(interval is not None and interval[0] <= 0 and interval[1] >= 1 for interval in intervals):
            result.append((a, b))
            continue
        cuts = sorted(set(cuts))
        pending = None
        for lo, hi in zip(cuts, cuts[1:]):
            if hi-lo <= 1e-12:
                continue
            eligible = {i for i, interval in enumerate(intervals)
                        if interval is not None and interval[0] <= lo+1e-12 and interval[1] >= hi-1e-12}
            if not eligible:
                point = a+d*((lo+hi)/2)
                raise ValueError(f'Unreachable edge {edge_index}: interval [{lo:.8f}, {hi:.8f}], XY={point.tolist()}, margin={margin_mm}mm')
            if pending is not None and pending[2]&eligible:
                pending = (pending[0], hi, pending[2]&eligible)
            else:
                if pending is not None:
                    result.append((a+d*pending[0], a+d*pending[1]))
                pending = (lo, hi, eligible)
        if pending is not None:
            result.append((a+d*pending[0], a+d*pending[1]))
    return result
