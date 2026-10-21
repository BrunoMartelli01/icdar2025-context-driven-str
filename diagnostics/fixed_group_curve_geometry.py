"""GT-free rectification of a FIXED component group, not a word detector."""
import cv2
import numpy as np


def bernstein(t):
    t = np.asarray(t)
    return np.stack(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3), axis=-1)


def sampling_map(spec):
    u = np.linspace(spec['u0'], spec['u1'], spec['width'])
    t = (u-spec['fit_u0'])/(spec['fit_u1']-spec['fit_u0'])
    # Constant edge extension: never extrapolate an unstable polynomial into padding.
    b = bernstein(np.clip(t, 0, 1))
    upper = b @ np.asarray(spec['upper']) - spec['pad']
    lower = b @ np.asarray(spec['lower']) + spec['pad']
    v = upper[None, :] + np.linspace(0, 1, spec['height'])[:, None]*(lower-upper)[None, :]
    axis = np.asarray(spec['axis']); normal = np.array([-axis[1], axis[0]])
    center = np.asarray(spec['center'])
    return ((center[0] + u[None, :]*axis[0] + v*normal[0]).astype(np.float32),
            (center[1] + u[None, :]*axis[1] + v*normal[1]).astype(np.float32))


def render(rgb, spec):
    mx, my = sampling_map(spec)
    return cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fit(group_mask, origin, image_shape, kind, cfg):
    """Only foreground from assigned raw CC; exclusive bbox origin in original pixels."""
    if kind not in ('ORIENTED', 'BEZIER'):
        raise ValueError('Unknown rectification')
    yy, xx = np.nonzero(group_mask)
    if len(xx) < cfg['minimum_pixels']:
        return None, 'insufficient_foreground'
    points = np.stack((xx+origin[0], yy+origin[1]), axis=1).astype(float)
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points-center, rowvar=False))
    if values[-1] <= 0 or np.isclose(values[-1], values[0]):
        return None, 'ambiguous_axis'
    axis = vectors[:, -1]
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    u = (points-center) @ axis; v = (points-center) @ normal
    lo, hi = float(u.min()), float(u.max())
    if hi-lo < 1:
        return None, 'degenerate_extent'
    t = (u-lo)/(hi-lo)
    if kind == 'ORIENTED':
        upper = np.full(4, v.min()-.5); lower = np.full(4, v.max()+.5)
    else:
        bins = np.minimum((t*cfg['envelope_bins']).astype(int), cfg['envelope_bins']-1)
        observations = [(float(t[bins == i].mean()), float(v[bins == i].min()), float(v[bins == i].max()))
                        for i in range(cfg['envelope_bins']) if np.any(bins == i)]
        if len(observations) < cfg['minimum_occupied_bins']:
            return None, 'sparse_longitudinal_support'
        obs = np.asarray(observations); basis = bernstein(obs[:, 0])
        if np.linalg.cond(basis) > cfg['max_condition_number']:
            return None, 'ill_conditioned_fit'
        upper = np.linalg.lstsq(basis, obs[:, 1], rcond=None)[0]
        lower = np.linalg.lstsq(basis, obs[:, 2], rcond=None)[0]
        upper -= max(0., float(np.max(bernstein(t) @ upper-v)))+.5
        lower += max(0., float(np.max(v-bernstein(t) @ lower)))+.5
    pad = cfg['pad']; grid = bernstein(np.linspace(0, 1, max(2, int(np.ceil(hi-lo))+1)))
    thickness = grid @ (lower-upper)
    if not np.isfinite(thickness).all() or thickness.min() <= 0:
        return None, 'crossing_envelopes'
    spec = dict(kind=kind, center=center.tolist(), axis=axis.tolist(), upper=upper.tolist(), lower=lower.tolist(),
                fit_u0=lo, fit_u1=hi, u0=lo-pad-.5, u1=hi+pad+.5, pad=pad,
                width=int(np.ceil(hi-lo+2*pad+1))+1, height=int(np.ceil(thickness.max()+2*pad))+1)
    mx, my = sampling_map(spec)
    h, w = image_shape[:2]
    if mx.min() < 0 or my.min() < 0 or mx.max() > w-1 or my.max() > h-1:
        return None, 'outside_image_fallback'
    spec['encloses_assigned_foreground'] = bool(np.all(bernstein(t) @ upper <= v) and np.all(bernstein(t) @ lower >= v))
    return spec, 'applied'
