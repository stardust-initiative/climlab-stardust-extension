"""Staggered symmetric tensor safeguards shared by atmospheric drivers.

Dyy is m2/s, Dpp Pa2/s and physical Dyp m Pa/s. Only the cross term is
reduced after interpolation; source diagonals are handled by the loader.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _collocate(values, x, y, tx, ty):
    xx, yy = np.meshgrid(np.clip(tx, x[0], x[-1]),
                         np.clip(ty, y[0], y[-1]), indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return RegularGridInterpolator((x, y), values, bounds_error=True)(points).reshape(xx.shape)


def bound_staggered_cross(yy, pp, cross, latc, latb, pc, pb):
    """Bound corner and both effective face cross coefficients by PSD limits.

    Cosine factors account for y=R*sin(latitude). Absolute-sum face bounds
    do not rely on cancellation of opposite-signed corner values. These
    bounds alone do not prove stability of the full mixed-flux discretization.
    """
    yy, pp, cross = [np.asarray(a) for a in (yy, pp, cross)]
    if not all(np.isfinite(a).all() for a in (yy, pp, cross)):
        raise ValueError("Diffusion tensor contains nonfinite values")
    if np.any(yy < 0) or np.any(pp < 0):
        raise ValueError("Diffusion diagonal must be nonnegative")
    yc = _collocate(yy, latb, pc, latb, pb)
    pcorner = _collocate(pp, latc, pb, latb, pb)
    corner_limit = np.sqrt(np.maximum(yc * pcorner, 0))
    limited = np.clip(cross, -corner_limit, corner_limit)
    cosine = np.cos(np.deg2rad(latb))[:, None]
    effective = limited * cosine
    yeffective = yy * cosine**2
    ybound = np.sin(np.deg2rad(latb))
    ycenter = .5 * (ybound[:-1] + ybound[1:])
    p_at_y = _collocate(pp, ycenter, pb, ybound, pc)
    y_at_p = _collocate(yeffective, ybound, pc, ycenter, pb)
    limit_y = np.sqrt(np.maximum(yeffective * p_at_y, 0))
    limit_p = np.sqrt(np.maximum(pp * y_at_p, 0))
    average_y = .5 * (abs(effective[:, :-1]) + abs(effective[:, 1:]))
    average_p = .5 * (abs(effective[:-1]) + abs(effective[1:]))
    fy = np.minimum(1, np.divide(limit_y, average_y, out=np.ones_like(limit_y), where=average_y > 0))
    fp = np.minimum(1, np.divide(limit_p, average_p, out=np.ones_like(limit_p), where=average_p > 0))
    scale = np.ones_like(limited)
    scale[:, :-1] = np.minimum(scale[:, :-1], fy)
    scale[:, 1:] = np.minimum(scale[:, 1:], fy)
    scale[:-1] = np.minimum(scale[:-1], fp)
    scale[1:] = np.minimum(scale[1:], fp)
    limited *= scale
    effective = limited * cosine
    def ratio(value, limit):
        if np.any((limit == 0) & (value != 0)):
            raise ValueError("Cross diffusion outside diagonal support")
        return float(np.max(np.divide(abs(value), limit, out=np.zeros_like(value), where=limit > 0)))
    stats = dict(max_corner_ratio=ratio(limited, corner_limit),
                 max_y_face_ratio=ratio(.5*(effective[:, :-1]+effective[:, 1:]), limit_y),
                 max_p_face_ratio=ratio(.5*(effective[:-1]+effective[1:]), limit_p),
                 extra_face_limited_corners=int((scale < 1).sum()))
    if max(stats[k] for k in ('max_corner_ratio','max_y_face_ratio','max_p_face_ratio')) > 1 + 1e-12:
        raise ValueError("Staggered tensor bound failed")
    return limited, stats
