"""Project-local methods: scene settings match stock vanilla-nerf."""
from dataclasses import dataclass, field
from typing import Literal, Type
import torch
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from .geometry import corrected_poses, preconditioners


@dataclass
class PoseModelConfig(VanillaModelConfig):
    _target: Type = field(default_factory=lambda: PoseModel)
    pose_method: Literal['naive', 'camp'] = 'naive'
    proxy_samples: int = 4096
    proxy_seed: int = 42


class PoseModel(NeRFModel):
    def populate_modules(self):
        super().populate_modules()
        self.pose_latent = torch.nn.Parameter(torch.zeros(self.num_train_data, 6))
        self.register_buffer('pose_preconditioner', torch.eye(6).repeat(self.num_train_data, 1, 1))
        self.register_buffer('initial_poses', torch.zeros(self.num_train_data, 3, 4))
        self.conditioning = []

    def physical_delta(self):
        return (self.pose_preconditioner @ self.pose_latent[..., None]).squeeze(-1)

    def optimized_poses(self):
        return corrected_poses(self.initial_poses, self.physical_delta())

    def get_param_groups(self):
        groups = super().get_param_groups()
        groups['camera_opt'] = [self.pose_latent]
        return groups

    def get_outputs(self, ray_bundle):
        # Only the training branch may index the training pose table.
        if self.training:
            from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
            indices = ray_bundle.camera_indices.reshape(-1).long()
            delta = self.physical_delta()[indices]
            correction = exp_map_SO3xR3(delta)
            ray_bundle.origins = ray_bundle.origins + delta[:, :3]
            ray_bundle.directions = (correction[:, :3, :3] @ ray_bundle.directions[..., None]).squeeze(-1)
        return super().get_outputs(ray_bundle)


@dataclass
class PosePipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: PosePipeline)


class PosePipeline(VanillaPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cameras = self.datamanager.train_dataset.cameras.to(self.device)
        self.model.initial_poses.copy_(cameras.camera_to_worlds)
        if self.model.config.pose_method == 'camp':
            collider = self.model.config.collider_params
            p, diagnostics = preconditioners(cameras, self.model.config.proxy_samples,
                                             self.model.config.proxy_seed,
                                             collider['near_plane'], collider['far_plane'])
            self.model.pose_preconditioner.copy_(p)
            self.model.conditioning = diagnostics


def specification(method):
    config = TrainerConfig(
        method_name='poseguard-'+method,
        pipeline=PosePipelineConfig(datamanager=VanillaDataManagerConfig(dataparser=BlenderDataParserConfig()),
                                    model=PoseModelConfig(pose_method=method)),
        optimizers={'fields': {'optimizer': RAdamOptimizerConfig(lr=5e-4, eps=1e-8), 'scheduler': None},
                    'temporal_distortion': {'optimizer': RAdamOptimizerConfig(lr=5e-4, eps=1e-8), 'scheduler': None},
                    'camera_opt': {'optimizer': AdamOptimizerConfig(lr=1e-3, eps=1e-8),
                                   'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000)}})
    return MethodSpecification(config=config, description='Vanilla NeRF with '+method+' pose optimization')


naive = specification('naive')
camp = specification('camp')
