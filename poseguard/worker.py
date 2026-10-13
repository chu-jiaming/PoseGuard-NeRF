"""Short-lived worker processes used by the durable serial orchestrator."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from .geometry import pose_errors
from .methods import naive, camp


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu')
    if not isinstance(state.get('step'), int) or not state.get('pipeline') or not state.get('optimizers'):
        raise ValueError('Incomplete checkpoint: '+str(path))
    if any(not torch.isfinite(value).all() for value in state['pipeline'].values() if torch.is_tensor(value)):
        raise ValueError('Nonfinite checkpoint tensor: '+str(path))
    return state


def train(args):
    from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation as Location
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    config = copy.deepcopy((naive if args.method == 'naive' else camp).config)
    config.output_dir = args.attempt / 'outputs'
    config.experiment_name = args.case
    config.timestamp = 'run'
    config.max_num_iterations = args.steps
    config.pipeline.datamanager.dataparser.data = args.data
    config.vis = 'tensorboard'
    config.viewer.quit_on_train_completion = True
    config.steps_per_save = args.interval
    camera_schedule = config.optimizers['camera_opt']['scheduler']
    camera_schedule.max_steps = args.steps
    camera_schedule.warmup_steps = args.camera_warmup_steps
    camera_schedule.lr_pre_warmup = 1e-8
    camera_schedule.ramp = 'cosine'
    if args.smoke:
        config.pipeline.datamanager.camera_res_scale_factor = .125
        config.pipeline.datamanager.train_num_rays_per_batch = 64
        config.pipeline.datamanager.eval_num_rays_per_batch = 64
        config.pipeline.model.num_coarse_samples = 16
        config.pipeline.model.num_importance_samples = 16
        config.pipeline.model.proxy_samples = 128
        config.steps_per_eval_image = 1000000
    config.load_checkpoint = args.checkpoint
    config.load_scheduler = True
    config.save_config()
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    model, pipeline = trainer.pipeline.model, trainer.pipeline
    if args.checkpoint:
        source = load_checkpoint(args.checkpoint)
        current = pipeline.state_dict()
        for name, value in source['pipeline'].items():
            if name not in current or not torch.equal(value.cpu(), current[name].detach().cpu()):
                raise ValueError('Checkpoint restore mismatch: '+name)
        if trainer._start_step != source['step']+1:
            raise ValueError('Incorrect resume start step')
        save(args.attempt/'resume_receipt.json', {'source': str(args.checkpoint),
             'source_step': source['step'], 'start_step': trainer._start_step,
             'remaining_steps': args.steps-trainer._start_step,
             'pipeline_tensors_verified': len(source['pipeline']),
             'camera_optimizer_restored': bool(trainer.optimizers.optimizers['camera_opt'].state),
             'camera_scheduler_last_epoch': trainer.optimizers.schedulers['camera_opt'].last_epoch})
    # GT is loaded AFTER constructing model and conditioning; diagnostics only.
    gt = json.loads((args.gt / 'transforms_train.json').read_text())
    reference = np.array([f['transform_matrix'][:3] for f in gt['frames']])
    save(args.attempt/'conditioning.json', model.conditioning)
    original_get_loss = pipeline.get_train_loss_dict
    latest = {}
    def tracked_loss(step):
        result = original_get_loss(step)
        if not all(torch.isfinite(value).all() for value in result[1].values()):
            raise FloatingPointError('Nonfinite training loss at step '+str(step))
        latest['loss'] = float(sum(result[1].values()).detach())
        latest['camera_lr'] = trainer.optimizers.optimizers['camera_opt'].param_groups[0]['lr']
        return result
    pipeline.get_train_loss_dict = tracked_loss
    trace_path = args.attempt/'diagnostics.jsonl'
    previous = model.physical_delta().detach().clone()
    def diagnose(step):
        nonlocal previous
        delta = model.physical_delta().detach()
        if not torch.isfinite(delta).all():
            raise FloatingPointError('Nonfinite camera parameters')
        # Diagnostic sampling must not alter the training random stream.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        pipeline.eval()
        try:
            with torch.random.fork_rng(), torch.no_grad():
                outputs, batch = pipeline.datamanager.next_eval(step)
                rendered = model(outputs)
                losses = model.get_loss_dict(rendered, batch)
                val_psnr = float(-10*torch.log10(losses['rgb_loss_fine'].clamp_min(1e-12)))
                error = pose_errors(model.optimized_poses().cpu().numpy(), reference)
        finally:
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            pipeline.train()
        row = dict(step=step, train_loss=latest.get('loss'), val_psnr=val_psnr,
                   camera_lr=latest.get('camera_lr'),
                   pose_gradient_norm=float(model.pose_latent.grad.norm()) if model.pose_latent.grad is not None else 0.,
                   translation_update_mean=float((delta[:, :3]-previous[:, :3]).norm(dim=-1).mean()),
                   rotation_update_deg_mean=float(torch.rad2deg((delta[:, 3:]-previous[:, 3:]).norm(dim=-1)).mean()),
                   pose_errors=error)
        with trace_path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
        previous = delta.clone()
        save(args.attempt/'optimized_poses.json', {'poses': model.optimized_poses().detach().cpu().tolist(), 'errors': error})
        print('POSE_DIAGNOSTIC', json.dumps(row), flush=True)
    trainer.callbacks.append(TrainingCallback([Location.AFTER_TRAIN_ITERATION], diagnose,
                                              update_every_num_iters=args.interval))
    trainer.callbacks.append(TrainingCallback([Location.AFTER_TRAIN], diagnose))
    # Stock 1.1.5 train() executes _start_step + max_num_iterations steps.
    remaining = args.steps-trainer._start_step
    if remaining <= 0:
        raise ValueError('Checkpoint already reaches target; use analysis stage')
    config.max_num_iterations = remaining
    trainer.train()
    save(args.attempt/'train_complete.json', {'step': trainer.step, 'target_steps': args.steps})


def analyze(args):
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image
    config, pipeline, checkpoint, step = eval_setup(args.config, test_mode='test')
    pipeline.eval()
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    model = pipeline.model
    if hasattr(model, 'optimized_poses'):
        gt = json.loads((args.gt/'transforms_train.json').read_text())
        reference = np.array([f['transform_matrix'][:3] for f in gt['frames']])
        with torch.no_grad():
            poses = model.optimized_poses().cpu().numpy()
        save(destination/'optimized_poses.json', {'poses': poses.tolist(), 'errors': pose_errors(poses, reference),
             'initial_errors': pose_errors(model.initial_poses.cpu().numpy(), reference)})
    selected = {0, 50, 100}
    if args.smoke:
        selected = {0}
    with torch.no_grad():
        for index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if index not in selected:
                continue
            outputs = model.get_outputs_for_camera(camera)
            for name in ('rgb', 'depth', 'accumulation'):
                array = outputs[name+'_fine'].cpu().numpy()
                if not np.isfinite(array).all():
                    raise FloatingPointError('Nonfinite render')
                np.save(destination/('view_%03d_%s.npy' % (index, name)), array)
                if name == 'depth':
                    near, far = model.config.collider_params['near_plane'], model.config.collider_params['far_plane']
                    array = np.clip((array-near)/(far-near), 0, 1)
                if array.shape[-1] == 1:
                    array = np.repeat(array, 3, axis=-1)
                Image.fromarray((np.clip(array, 0, 1)*255).astype('uint8')).save(destination/('view_%03d_%s.png' % (index, name)))
            if index == max(selected):
                break
    save(destination/'analysis_complete.json', {'checkpoint': str(checkpoint), 'step': step, 'views': sorted(selected)})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--method', choices=['naive', 'camp'], required=True)
    training.add_argument('--case', required=True)
    training.add_argument('--data', type=Path, required=True)
    training.add_argument('--attempt', type=Path, required=True)
    training.add_argument('--steps', type=int, required=True)
    training.add_argument('--camera-warmup-steps', type=int, default=0)
    training.add_argument('--interval', type=int, default=1000)
    training.add_argument('--checkpoint', type=Path)
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--config', type=Path, required=True)
    analysis.add_argument('--destination', type=Path, required=True)
    for item in (training, analysis):
        item.add_argument('--gt', type=Path, required=True)
        item.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'train' and not 0 <= args.camera_warmup_steps < args.steps:
        parser.error('camera-warmup-steps must be nonnegative and smaller than steps')
    (train if args.command == 'train' else analyze)(args)


if __name__ == '__main__':
    main()
