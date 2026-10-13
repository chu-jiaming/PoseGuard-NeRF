#!/usr/bin/env python
"""Durable sequential pose experiments; all GPU stages are separate processes."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sv100_pipeline import CASES, DATA, GT, validate
from poseguard.worker import save, load_checkpoint


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log_path):
    print(now(), 'COMMAND', shlex.join(map(str, command)), 'LOG', log_path, flush=True)
    with Path(log_path).open('a') as output:
        output.write(now()+' '+shlex.join(map(str, command))+'\n'); output.flush()
        process = subprocess.Popen(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            rc = process.wait()
        except BaseException:
            if process.poll() is not None:
                raise
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    if rc:
        raise RuntimeError('Worker exit %s: %s' % (rc, log_path))


def preflight():
    if importlib.metadata.version('nerfstudio') != '1.1.5':
        raise RuntimeError('Requires Nerfstudio 1.1.5')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    print('GPU:', torch.cuda.get_device_name(), flush=True)
    if subprocess.run(['pgrep', '-f', r'(^|/|[[:space:]])ns-train([[:space:]]|$)'], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('An external ns-train process is active')
    manifest = {case[0]: validate(case) for case in CASES}
    with (ROOT/'results/SV100_full_summary.csv').open() as f:
        baselines = {row['case']: row for row in csv.DictReader(f)}
    fingerprint = {}
    for case in CASES:
        row = baselines[case[0]]
        if row['status'] != 'SUCCESS' or int(row['max_iterations']) != 30000:
            raise RuntimeError('Incomplete baseline '+case[0])
        for key in ('config_path', 'checkpoint_path', 'evaluation_result'):
            path = Path(row[key])
            if not path.is_file():
                raise RuntimeError('Missing baseline artifact '+str(path))
            fingerprint[str(path)] = digest(path)
        metrics = read(row['evaluation_result'])
        if Path(metrics['checkpoint']).resolve() != Path(row['checkpoint_path']).resolve():
            raise RuntimeError('Baseline checkpoint mismatch')
        if not all(math.isfinite(float(metrics['results']['fine_'+key])) for key in ('psnr', 'ssim', 'lpips')):
            raise RuntimeError('Invalid baseline metrics')
    code = {str(path.relative_to(ROOT)): digest(path) for path in
            sorted(list((ROOT/'poseguard').glob('*.py')) + [Path(__file__).resolve(), ROOT/'scripts/sv100_pipeline.py',
                                                           ROOT/'scripts/run_pose_optimization_pipeline.sh'])}
    return baselines, {'data': manifest, 'baseline': fingerprint, 'code': code,
                       'python': sys.version, 'nerfstudio': '1.1.5', 'torch': torch.__version__,
                       'packages': {name: importlib.metadata.version(name) for name in
                                    ('numpy', 'scipy', 'Pillow', 'matplotlib', 'torchmetrics', 'tyro')}}


def config_for(attempt):
    configs = list((attempt/'outputs').rglob('config.yml'))
    if len(configs) != 1:
        raise RuntimeError('Config not uniquely identified in '+str(attempt))
    return configs[0]


def verify_config(config_path, method, case, args):
    import yaml
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    dm = config.pipeline.datamanager
    data = dm.data if dm.data is not None else dm.dataparser.data
    if (config.method_name != 'poseguard-'+method or config.pipeline.model.pose_method != method or
            config.max_num_iterations != args.steps or config.machine.seed != 42 or
            Path(data).resolve() != (DATA/case[3]).resolve()):
        raise ValueError('Saved config does not match experiment')
    schedule = config.optimizers['camera_opt']['scheduler']
    if (schedule.warmup_steps != args.camera_warmup_steps or schedule.max_steps != args.steps
            or schedule.lr_pre_warmup != 1e-8 or schedule.ramp != 'cosine'):
        raise ValueError('Saved camera schedule mismatch')
    if config.pipeline.datamanager.camera_res_scale_factor != (.125 if args.smoke else 1.):
        raise ValueError('Saved resolution mismatch')


def latest_checkpoint(attempt, target):
    for path in sorted((attempt/'outputs').rglob('step-*.ckpt'), reverse=True):
        try:
            checkpoint = load_checkpoint(path)
            step = checkpoint['step']
            if step < 0 or step >= target:
                raise ValueError('Checkpoint outside training target')
            if not all(name in checkpoint['pipeline'] for name in
                       ('_model.pose_latent', '_model.pose_preconditioner', '_model.initial_poses')):
                raise ValueError('Missing pose buffers')
            if 'camera_opt' not in checkpoint['optimizers'] or 'camera_opt' not in checkpoint.get('schedulers', {}):
                raise ValueError('Missing camera optimizer/scheduler state')
            if checkpoint['pipeline']['_model.pose_latent'].shape != (100, 6):
                raise ValueError('Unexpected camera count')
            return path, step
        except Exception as error:
            print('INVALID CHECKPOINT', path, str(error), flush=True)
    return None, -1


def worker(*args):
    return [sys.executable, '-m', 'poseguard.worker', *args]


def evaluate_metrics(path, checkpoint):
    data = read(path)
    if Path(data['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Evaluation used a different checkpoint')
    metrics = {key: float(data['results']['fine_'+key]) for key in ('psnr', 'ssim', 'lpips')}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite evaluation metric')
    return metrics


def valid_analysis(attempt, checkpoint, step, smoke, poses=True):
    try:
        result = read(attempt/'analysis_complete.json')
        if result['step'] != step or Path(result['checkpoint']).resolve() != checkpoint.resolve():
            return False
        for view in ([0] if smoke else [0, 50, 100]):
            for kind in ('rgb', 'depth', 'accumulation'):
                from PIL import Image
                with Image.open(attempt/('view_%03d_%s.png' % (view, kind))) as im:
                    im.verify()
                if not (attempt/('view_%03d_%s.npy' % (view, kind))).is_file():
                    return False
        if poses:
            import numpy as np
            data = read(attempt/'optimized_poses.json')
            array = np.asarray(data['poses'])
            if array.shape != (100, 3, 4) or not np.isfinite(array).all():
                return False
            if not data.get('errors'):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def execute_case(args, case, method, entry, persist):
    name, kind, magnitude, directory = case
    attempts = [Path(x) for x in entry.get('attempts', [])]
    checkpoint, step, source_attempt = None, -1, None
    for candidate in reversed(attempts):
        found, found_step = latest_checkpoint(candidate, args.steps)
        if found_step > step:
            checkpoint, step, source_attempt = found, found_step, candidate
    if step == args.steps-1:
        attempt = source_attempt
    else:
        attempt = args.run_dir/'experiments'/method/name/('attempt_%03d' % (len(attempts)+1))
        attempt.mkdir(parents=True)
        entry.setdefault('attempts', []).append(str(attempt))
        entry.update(stage='train', status='RUNNING'); persist()
        command = worker('train', '--method', method, '--case', name, '--data', DATA/directory,
                         '--attempt', attempt, '--steps', args.steps, '--interval', args.interval, '--gt', GT,
                         '--camera-warmup-steps', args.camera_warmup_steps)
        if checkpoint:
            command += ['--checkpoint', checkpoint]
        if args.smoke:
            command += ['--smoke']
        run(command, attempt/'train.log')
        checkpoint, step = latest_checkpoint(attempt, args.steps)
        if step != args.steps-1:
            raise RuntimeError('Training did not reach target step')
    config = config_for(attempt)
    verify_config(config, method, case, args)
    entry.update(config=str(config), checkpoint=str(checkpoint), attempt=str(attempt), stage='eval'); persist()
    metrics_path = attempt/'metrics.json'
    if metrics_path.exists():
        try:
            metrics = evaluate_metrics(metrics_path, checkpoint)
        except Exception:
            metrics_path.rename(attempt/('invalid_metrics_'+datetime.now().strftime('%H%M%S')+'.json'))
    if not metrics_path.exists():
        run(['ns-eval', '--load-config', config, '--output-path', metrics_path], attempt/'eval.log')
    metrics = evaluate_metrics(metrics_path, checkpoint)
    entry.update(stage='analyze', metrics=metrics); persist()
    if not valid_analysis(attempt, checkpoint, args.steps-1, args.smoke):
        command = worker('analyze', '--config', config, '--destination', attempt, '--gt', GT)
        if args.smoke:
            command += ['--smoke']
        run(command, attempt/'analyze.log')
    if not valid_analysis(attempt, checkpoint, args.steps-1, args.smoke):
        raise ValueError('Analysis artifacts invalid')
    entry.update(status='SUCCESS', stage='complete', end=now(), metrics=metrics, error=None)
    persist()


def report(args, state, baselines, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    images = args.run_dir/'figures'; images.mkdir(exist_ok=True)
    lines = ['# Pose optimization — '+label, '',
             'Camera warm-up: %d steps; pre-warmup LR 1e-8; Nerfstudio cosine ramp.' % args.camera_warmup_steps,
             'Single seed 42; vanilla-NeRF; target %d steps. Fixed baselines: 30000 steps.' % args.steps,
             'Smoke results are pipeline checks, not comparable research results.' if args.smoke else
             'Test cameras remain in original coordinates. Sim(3) alignment is used only for pose diagnostics.', '',
             '| Case | Method | Status | PSNR | SSIM | LPIPS | Raw rot ° | Aligned rot ° |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    details = []
    for case in args.selected:
        name = case[0]
        baseline = baselines[name]
        rows.append(dict(case=name, method='fixed', status='SUCCESS',
                         **{key: float(baseline[key]) for key in ('psnr', 'ssim', 'lpips')}))
        for method in ('naive', 'camp'):
            entry = state['experiments'].get(method+'/'+name, {})
            row = dict(case=name, method=method, status=entry.get('status', 'PENDING'), **entry.get('metrics', {}))
            if entry.get('status') == 'SUCCESS':
                error = read(Path(entry['attempt'])/'optimized_poses.json')['errors']
                row.update(raw_rotation_deg=error['raw']['rotation_deg_mean'],
                           raw_translation_pct=error['raw']['translation_radius_pct_mean'],
                           aligned_rotation_deg=error['aligned']['rotation_deg_mean'] if error['aligned'] else None,
                           aligned_translation_pct=error['aligned']['translation_radius_pct_mean'] if error['aligned'] else None,
                           psnr_delta_fixed=row['psnr']-float(baseline['psnr']))
            row['error'] = entry.get('error', '')
            rows.append(row)
        for row in rows[-3:]:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % tuple(
                row.get(key, '—') for key in ('case', 'method', 'status', 'psnr', 'ssim', 'lpips', 'raw_rotation_deg', 'aligned_rotation_deg')))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.flatten()
        notes = []
        for method in ('naive', 'camp'):
            entry = state['experiments'].get(method+'/'+name, {})
            trace = {}
            for attempt in entry.get('attempts', []):
                path = Path(attempt)/'diagnostics.jsonl'
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:
                            item = json.loads(line); trace[item['step']] = item
                        except json.JSONDecodeError:
                            continue  # a process may die mid-write
            values = [trace[step] for step in sorted(trace)]
            if values:
                x = [v['step'] for v in values]
                for ax, y, title in zip(axes,
                    ([v['train_loss'] for v in values], [v['val_psnr'] for v in values],
                     [v['pose_errors']['raw']['rotation_deg_mean'] for v in values],
                     [v['pose_errors']['raw']['translation_radius_pct_mean'] for v in values],
                     [v['pose_gradient_norm'] for v in values], [v['rotation_update_deg_mean'] for v in values]),
                    ('Train loss', 'Validation batch PSNR', 'Raw rotation error (deg)',
                     'Raw translation error (% radius)', 'Latent camera gradient norm', 'Rotation update (deg / interval)')):
                    ax.plot(x, y, label=method); ax.set_title(title); ax.set_xlabel('step'); ax.legend()
                if len(values) >= 4 and values[-1]['val_psnr']-values[len(values)//2]['val_psnr'] < .1:
                    notes.append(method+': late validation stagnation/decline candidate (batch-sampling noise possible).')
                if len(values) >= 4:
                    differences = [values[i]['val_psnr']-values[i-1]['val_psnr'] for i in range(1, len(values))]
                    if sum(a*b < 0 for a, b in zip(differences, differences[1:])) >= 2:
                        notes.append(method+': oscillation candidate; inspect curves and camera updates.')
            if entry.get('status') == 'FAILED':
                notes.append(method+': runtime failure: '+str(entry.get('error')))
            if entry.get('status') == 'SUCCESS' and entry['metrics']['psnr'] < float(baseline['psnr']):
                notes.append(method+': test PSNR is below the fixed-pose baseline.')
            if len(values) >= 2 and values[-1]['val_psnr'] > values[0]['val_psnr'] and (
                    values[-1]['pose_errors']['raw']['rotation_deg_mean'] > values[0]['pose_errors']['raw']['rotation_deg_mean'] or
                    values[-1]['pose_errors']['raw']['translation_mean'] > values[0]['pose_errors']['raw']['translation_mean']):
                notes.append(method+': image/pose metrics diverge; scene-camera compensation is a hypothesis, not a proven cause.')
        fig.tight_layout(); fig.savefig(images/(name+'_curves.png')); plt.close(fig)
        details += ['', '## '+name, '', '[Convergence curves](figures/'+name+'_curves.png)', '']
        details += notes or ['No automatic failure flag; inspect images and pose diagnostics.']
        for view in ([0] if args.smoke else [0, 50, 100]):
            for kind in ('rgb', 'depth', 'accumulation'):
                filename = '%s_view_%03d_%s.png' % (name, view, kind)
                if (images/filename).exists():
                    details += ['', '[View %d %s comparison](figures/%s)' % (view, kind, filename)]
    fields = ['case', 'method', 'status', 'psnr', 'ssim', 'lpips', 'psnr_delta_fixed', 'raw_rotation_deg',
              'aligned_rotation_deg', 'raw_translation_pct', 'aligned_translation_pct', 'error']
    temporary = args.run_dir/'summary.csv.tmp'
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(args.run_dir/'summary.csv')
    for error in state.get('baseline_render_errors', []):
        details += ['', 'Baseline visualization failure: '+error]
    lines += details
    (args.run_dir/'report.md').write_text('\n'.join(lines)+'\n')


def comparisons(args, state, baselines, persist):
    from PIL import Image, ImageDraw
    for case in args.selected:
        name = case[0]
        baseline = args.run_dir/'fixed_renders'/name
        baseline.mkdir(parents=True, exist_ok=True)
        try:
            if not valid_analysis(baseline, Path(baselines[name]['checkpoint_path']), 29999, args.smoke, poses=False):
                command = worker('analyze', '--config', baselines[name]['config_path'], '--destination', baseline, '--gt', GT)
                if args.smoke:
                    command += ['--smoke']
                run(command, baseline/'analyze.log')
            for view in ([0] if args.smoke else [0, 50, 100]):
                for kind in ('rgb', 'depth', 'accumulation'):
                    paths = [('fixed', baseline/('view_%03d_%s.png' % (view, kind)))]
                    for method in ('naive', 'camp'):
                        entry = state['experiments'].get(method+'/'+name, {})
                        if entry.get('status') == 'SUCCESS':
                            paths.append((method, Path(entry['attempt'])/('view_%03d_%s.png' % (view, kind))))
                    tiles = [(title, Image.open(path).convert('RGB')) for title, path in paths]
                    # Smoke training uses smaller images; rescale ONLY the comparison preview.
                    width, height = tiles[0][1].size
                    canvas = Image.new('RGB', (width*len(tiles), height+25), 'white')
                    draw = ImageDraw.Draw(canvas)
                    for i, (title, image) in enumerate(tiles):
                        canvas.paste(image.resize((width, height)), (i*width, 25)); draw.text((i*width+5, 5), title, fill='black')
                    canvas.save(args.run_dir/'figures'/('%s_view_%03d_%s.png' % (name, view, kind)))
        except Exception as error:
            state.setdefault('baseline_render_errors', []).append(name+': '+str(error)); persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'results/poseopt_sv100')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--camera-warmup-steps', type=int, default=0)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in CASES])
    args = parser.parse_args()
    if args.steps <= 0 or args.interval <= 0:
        parser.error('steps and interval must be positive')
    if not 0 <= args.camera_warmup_steps < args.steps:
        parser.error('camera-warmup-steps must be nonnegative and smaller than steps')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    if args.smoke and args.run_dir == ROOT/'results/poseopt_sv100':
        parser.error('--smoke requires a separate --run-dir')
    args.run_dir = args.run_dir.resolve()
    args.selected = [c for c in CASES if not args.cases or c[0] in args.cases]
    baselines, manifest = preflight()
    if args.preflight:
        print('PREFLIGHT OK'); return 0
    if args.dry_run:
        for method in ('naive', 'camp'):
            for case in args.selected:
                print(method, case[0], DATA/case[3], 'target_steps='+str(args.steps))
        print('No training or results written.'); return 0
    state_path = args.run_dir/'state.json'
    settings = dict(steps=args.steps, interval=args.interval, smoke=args.smoke,
                    camera_warmup_steps=args.camera_warmup_steps,
                    order=[method+'/'+c[0] for method in ('naive', 'camp') for c in args.selected])
    if state_path.exists():
        if not args.resume:
            raise RuntimeError('Run exists; use --resume or choose another --run-dir')
        state = read(state_path)
        if state['settings'] != settings or read(args.run_dir/'manifest.json') != manifest:
            raise RuntimeError('Configuration/code/data/environment fingerprints changed; choose a new run-dir')
    else:
        if args.resume:
            raise RuntimeError('No run to resume')
        args.run_dir.mkdir(parents=True, exist_ok=False)
        state = dict(status='RUNNING', start=now(), settings=settings, experiments={})
        save(args.run_dir/'manifest.json', manifest)
        run([sys.executable, '-m', 'pip', 'freeze'], args.run_dir/'environment.txt')
    class Tee:
        def __init__(self, stream, handle):
            self.stream = stream
            self.handle = handle
        def write(self, value):
            self.stream.write(value); self.handle.write(value)
        def flush(self):
            self.stream.flush(); self.handle.flush()
        def isatty(self):
            return False
    original_streams = sys.stdout, sys.stderr
    log_handle = (args.run_dir/'pipeline.log').open('a', buffering=1)
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    state.update(status='RUNNING', pid=os.getpid(), last_start=now(), baseline_render_errors=[])
    state.pop('error', None)
    state.pop('end', None)
    def persist():
        state['updated'] = now(); save(state_path, state)
    persist()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    active = None
    try:
        for method in ('naive', 'camp'):
            for case in args.selected:
                key = method+'/'+case[0]
                active = state['experiments'].setdefault(key, dict(status='PENDING', attempts=[]))
                if active['status'] == 'FAILED' and not args.retry_failed:
                    continue
                if active['status'] == 'SUCCESS':
                    attempt = Path(active['attempt'])
                    checkpoint, step = latest_checkpoint(attempt, args.steps)
                    if step == args.steps-1 and checkpoint and valid_analysis(attempt, checkpoint, step, args.smoke):
                        try:
                            verify_config(config_for(attempt), method, case, args)
                            evaluate_metrics(attempt/'metrics.json', checkpoint)
                            print('SKIP verified SUCCESS', key, flush=True); continue
                        except Exception:
                            pass
                active.update(status='RUNNING', start=now(), error=None); persist()
                try:
                    if {c[0]: validate(c) for c in [case]}[case[0]] != manifest['data'][case[0]]:
                        raise RuntimeError('Data changed after preflight')
                    execute_case(args, case, method, active, persist)
                except Exception as error:
                    active.update(status='FAILED', error=str(error), end=now()); persist()
                    print('FAILED', key, str(error), flush=True)
                    (args.run_dir/('failure_'+key.replace('/', '_')+'.txt')).write_text(traceback.format_exc())
                report(args, state, baselines, 'RUNNING')
            report(args, state, baselines, method+' phase complete')
            if method == 'naive':
                (args.run_dir/'naive_failure_analysis.md').write_text((args.run_dir/'report.md').read_text())
        comparisons(args, state, baselines, persist)
        state['status'] = 'FAILED' if state.get('baseline_render_errors') or any(
            item['status'] != 'SUCCESS' for item in state['experiments'].values()) else 'SUCCESS'
        state['end'] = now(); persist()
        report(args, state, baselines, state['status'])
        print(now(), state['status'], 'Report:', args.run_dir/'report.md', flush=True)
        return 0 if state['status'] == 'SUCCESS' else 1
    except BaseException as error:
        if active and active['status'] == 'RUNNING':
            active.update(status='INTERRUPTED', error=str(error))
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        persist()
        raise
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = original_streams
        log_handle.close()


if __name__ == '__main__':
    os.chdir(ROOT)
    sys.exit(main())
