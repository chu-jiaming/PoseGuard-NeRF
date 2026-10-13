import unittest
import numpy as np
from poseguard.alignment import fit_sim3, transform_poses

class AlignmentTests(unittest.TestCase):
    def test_known_transform_and_projection_invariance(self):
        rng = np.random.RandomState(123)
        poses = np.tile(np.eye(4)[:3], (20, 1, 1))
        poses[:, :, 3] = rng.randn(20, 3)
        known = dict(scale=1.7, rotation=[[0,-1,0],[1,0,0],[0,0,1]], translation=[1,2,3])
        target = transform_poses(poses, known)
        fit = fit_sim3(poses[:,:,3], target[:,:,3])
        np.testing.assert_allclose(transform_poses(poses, fit), target, atol=1e-12)
        point = np.array([.1, .3, -2.])
        transformed = known['scale'] * np.asarray(known['rotation']) @ point + known['translation']
        q = poses[0,:,:3].T @ (point-poses[0,:,3])
        qt = target[0,:,:3].T @ (transformed-target[0,:,3])
        np.testing.assert_allclose(q[:2]/q[2], qt[:2]/qt[2], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(qt), known['scale']*np.linalg.norm(q))

    def test_identity_and_degenerate(self):
        centers = np.random.RandomState(42).randn(20,3)
        fit = fit_sim3(centers, centers)
        np.testing.assert_allclose(fit['rotation'], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(fit['scale'], 1.)
        with self.assertRaises(ValueError):
            fit_sim3(np.zeros((4,3)), np.zeros((4,3)))
