"""Evaluation-only global similarity fit; no image-based optimization."""
import numpy as np


def fit_sim3(source, target):
    """Fit target = scale * rotation @ source + translation, using camera centers."""
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Expected matching N x 3 centers, N >= 3')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite centers')
    xc, yc = x-x.mean(0), y-y.mean(0)
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError('Degenerate camera layout')
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3); sign[-1] = np.linalg.det(u @ vt)
    rotation = (u*sign) @ vt
    scale = float((singular*sign).sum() / (xc*xc).sum() * len(x))
    translation = y.mean(0)-scale*rotation @ x.mean(0)
    if scale <= 0:
        raise ValueError('Invalid scale')
    return dict(scale=scale, rotation=rotation.tolist(), translation=translation.tolist())


def transform_poses(poses, sim3):
    result = np.array(poses, dtype=np.float64, copy=True)
    r, t = np.asarray(sim3['rotation']), np.asarray(sim3['translation'])
    result[..., :3, :3] = r @ result[..., :3, :3]
    result[..., :3, 3] = sim3['scale'] * result[..., :3, 3] @ r.T + t
    return result
