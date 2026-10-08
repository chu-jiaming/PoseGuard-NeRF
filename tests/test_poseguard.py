import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.vanilla_nerf import NeRFModel
from poseguard.geometry import corrected_poses, project, projection_jacobian, preconditioners, pose_errors
from poseguard.methods import PoseModel


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.pose = torch.eye(4, dtype=torch.float64)[:3]
        self.pose[:, 3] = torch.tensor([.3, -.2, 2.])
        self.points = torch.tensor([[.1, .2, -.5], [-.4, .1, -1.]], dtype=torch.float64)

    def test_projection_derivative_matches_finite_difference(self):
        analytic = projection_jacobian(self.points, self.pose, (.9, .8))
        numerical = torch.zeros_like(analytic)
        for k in range(6):
            delta = torch.zeros(1, 6, dtype=torch.float64); delta[0, k] = 1e-5
            plus = project(self.points, corrected_poses(self.pose[None], delta)[0], (.9, .8))
            minus = project(self.points, corrected_poses(self.pose[None], -delta)[0], (.9, .8))
            numerical[:, :, k] = (plus-minus)/2e-5
        torch.testing.assert_close(analytic, numerical, atol=2e-5, rtol=2e-5)

    def test_identity_and_known_world_correction(self):
        torch.testing.assert_close(corrected_poses(self.pose[None], torch.zeros(1, 6, dtype=torch.float64))[0], self.pose)
        delta = torch.tensor([[1., 2., 3., 0., 0., np.pi/2]], dtype=torch.float64)
        updated = corrected_poses(self.pose[None], delta)[0]
        torch.testing.assert_close(updated[:, 3], self.pose[:, 3]+delta[0, :3])
        torch.testing.assert_close(updated[:, :3], torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64), atol=1e-12, rtol=0)

    def test_conditioning_finite_deterministic_isolated_rng(self):
        cameras = Cameras(camera_to_worlds=self.pose[None].float(), fx=600., fy=600., cx=400., cy=400., width=800, height=800)
        state = torch.random.get_rng_state().clone()
        p, diagnostics = preconditioners(cameras, samples=512)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        p2, _ = preconditioners(cameras, samples=512)
        torch.testing.assert_close(p, p2)
        self.assertTrue(torch.isfinite(p).all())
        self.assertLess(diagnostics[0]['condition_after'], diagnostics[0]['condition_before'])
        self.assertGreater(float(p[0].abs().sum()-p[0].diag().abs().sum()), .01)

    def test_sim3_alignment_removes_global_gauge(self):
        rng = np.random.RandomState(42)
        reference = np.tile(np.eye(4)[:3], (20, 1, 1))
        reference[:, :3, 3] = rng.randn(20, 3)
        estimated = reference.copy()
        r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        estimated[:, :3, :3] = r @ reference[:, :3, :3]
        estimated[:, :3, 3] = 2*reference[:, :3, 3] @ r.T + np.array([1, 2, 3])
        error = pose_errors(estimated, reference)
        self.assertGreater(error['raw']['rotation_deg_mean'], 80)
        self.assertLess(error['aligned']['rotation_deg_mean'], 1e-5)
        self.assertLess(error['aligned']['translation_mean'], 1e-12)

    def model(self):
        model = PoseModel.__new__(PoseModel)
        torch.nn.Module.__init__(model)
        model.pose_latent = torch.nn.Parameter(torch.zeros(2, 6))
        model.register_buffer('pose_preconditioner', torch.eye(6).repeat(2, 1, 1))
        return model

    def ray(self, index=0):
        return RayBundle(origins=torch.zeros(1, 3), directions=torch.tensor([[1., 0., 0.]]),
                         pixel_area=torch.ones(1, 1), camera_indices=torch.tensor([[index]]))

    def test_training_gradient_and_eval_isolation(self):
        model = self.model()
        with patch.object(NeRFModel, 'get_outputs', side_effect=lambda rays: rays):
            ray = model.get_outputs(self.ray())
            torch.testing.assert_close(ray.origins, torch.zeros(1, 3))
            loss = ray.origins.sum()+ray.directions[:, 1].sum()
            loss.backward()
            self.assertGreater(float(model.pose_latent.grad.abs().sum()), 0)
            optimizer = torch.optim.Adam([model.pose_latent], lr=.001)
            optimizer.step()
            self.assertGreater(float(model.pose_latent.detach().abs().sum()), 0)
            model.eval()
            # Held-out indices need not fit the train camera table.
            ray = model.get_outputs(self.ray(index=199))
            torch.testing.assert_close(ray.origins, torch.zeros(1, 3))
            torch.testing.assert_close(ray.directions, torch.tensor([[1., 0., 0.]]))

    def test_pose_checkpoint_buffers_round_trip(self):
        first = self.model()
        with torch.no_grad():
            first.pose_latent.normal_(); first.pose_preconditioner.mul_(2)
        second = self.model()
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        torch.testing.assert_close(first.physical_delta(), second.physical_delta())


if __name__ == '__main__':
    unittest.main()
