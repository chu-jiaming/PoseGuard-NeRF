"""World-frame SO3 x R3 corrections and projection-proxy conditioning."""
import numpy as np
import torch
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3


def corrected_poses(initial, delta):
    correction = exp_map_SO3xR3(delta)
    return torch.cat((correction[:, :3, :3] @ initial[:, :3, :3],
                      initial[:, :3, 3:] + delta[:, :3, None]), dim=-1)


def skew(x):
    a, b, c = x.unbind(-1)
    zero = torch.zeros_like(a)
    return torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), -1).reshape(*x.shape[:-1], 3, 3)


def project(points, pose, focal):
    """Normalized pixels (u-cx)/width, (v-cy)/height; OpenGL camera axes."""
    q = (points - pose[:, 3]) @ pose[:, :3]
    return torch.stack((focal[0] * q[:, 0] / -q[:, 2],
                        -focal[1] * q[:, 1] / -q[:, 2]), -1)


def projection_jacobian(points, pose, focal):
    relative = points - pose[:, 3]
    q = relative @ pose[:, :3]
    x, y, z = q.unbind(-1)
    zero = torch.zeros_like(z)
    d = torch.stack((-focal[0]/z, zero, focal[0]*x/z.square(),
                     zero, focal[1]/z, -focal[1]*y/z.square()), -1).reshape(-1, 2, 3)
    rt = pose[:, :3].T.expand(len(points), -1, -1)
    return d @ torch.cat((-rt, rt @ skew(relative)), dim=-1)


def preconditioners(cameras, samples=4096, seed=42, near=2., far=6.):
    """Fixed damped ZCA transform. Uses an isolated CPU generator, no scene/GT."""
    rng = torch.Generator().manual_seed(seed)
    result, diagnostics = [], []
    for i, pose in enumerate(cameras.camera_to_worlds.detach().cpu().double()):
        fx, fy, cx, cy, w, h = [float(getattr(cameras, key).flatten()[i])
                                for key in ('fx', 'fy', 'cx', 'cy', 'width', 'height')]
        uvd = torch.rand(samples, 3, generator=rng, dtype=torch.float64)
        directions = torch.stack(((uvd[:, 0]*w-cx)/fx, -(uvd[:, 1]*h-cy)/fy,
                                  -torch.ones(samples, dtype=torch.float64)), -1)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        points = (directions * (near+(far-near)*uvd[:, 2:])) @ pose[:, :3].T + pose[:, 3]
        j = projection_jacobian(points, pose, (fx/w, fy/h)).reshape(-1, 6)
        sigma = j.T @ j / samples
        damped = sigma + .1*torch.diag(sigma.diag()) + 1e-8*torch.eye(6, dtype=torch.float64)
        eig, vectors = torch.linalg.eigh(damped)
        if not torch.isfinite(eig).all() or (eig <= 0).any():
            raise ValueError('Invalid projection covariance')
        p = (vectors * eig.rsqrt()[None, :]) @ vectors.T
        result.append(p.float())
        diagnostics.append({'camera': i, 'condition_before': float(torch.linalg.cond(sigma)),
                            'condition_after': float(torch.linalg.cond(p.T @ sigma @ p))})
    return torch.stack(result).to(cameras.device), diagnostics


def pose_errors(estimated, reference):
    """Raw and camera-center Umeyama Sim(3)-aligned error, evaluation only."""
    a, b = np.asarray(estimated), np.asarray(reference)
    radius = np.linalg.norm(b[:, :3, 3], axis=-1).mean()
    def errors(x):
        rotation = x[:, :3, :3] @ b[:, :3, :3].transpose(0, 2, 1)
        angles = np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2)-1)/2, -1, 1)))
        distance = np.linalg.norm(x[:, :3, 3]-b[:, :3, 3], axis=-1)
        return {'rotation_deg_mean': float(angles.mean()), 'rotation_deg_max': float(angles.max()),
                'translation_mean': float(distance.mean()), 'translation_radius_pct_mean': float(100*distance.mean()/radius)}
    x, y = a[:, :3, 3], b[:, :3, 3]
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, s, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3); sign[-1] = np.linalg.det(u @ vt)
    r = (u * sign) @ vt
    variance = np.square(xc).sum()/len(x)
    if variance < 1e-12:
        return {'raw': errors(a), 'aligned': None, 'alignment_error': 'degenerate camera centers'}
    scale = float((s*sign).sum()/variance)
    t = y.mean(0)-scale*r @ x.mean(0)
    aligned = a.copy()
    aligned[:, :3, 3] = scale*x @ r.T+t
    aligned[:, :3, :3] = r @ a[:, :3, :3]
    return {'raw': errors(a), 'aligned': errors(aligned),
            'sim3': {'scale': scale, 'rotation': r.tolist(), 'translation': t.tolist()}}
