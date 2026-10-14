#!/usr/bin/env python
"""Build the README figures from archived metrics and a server showcase delivery.

Metric figures read artifacts/ only. Render figures and the GIF also need the
extracted delivery_<date>/ directory (800x800 stills, 400x400 video); pass it with
--delivery or they are skipped. The GIF needs ffmpeg on PATH. The camera figure
needs the public NeRF-Synthetic lego/transforms_train.json (--gt-transforms).
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT/'artifacts/analysis'
CONDITIONS = ['SV100-GT', 'SV100-T0.5', 'SV100-T1', 'SV100-T2', 'SV100-T5',
              'SV100-R0.5', 'SV100-R1', 'SV100-R2', 'SV100-R5']
METHODS = ['fixed', 'naive', 'camp', 'naive_warmup', 'camp_warmup']
LABELS = {'fixed': 'Fixed cameras', 'naive': 'Naive', 'camp': 'CamP-style',
          'naive_warmup': 'Naive + warm-up', 'camp_warmup': 'CamP-style + warm-up'}
# Validated categorical pair (blue, orange); gray is the neutral fixed-camera reference.
INK, MUTED, SURFACE, GRID = '#0b0b0b', '#52514e', '#ffffff', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def short(condition):
    """SV100-T0.5 -> 'translation 0.5%', SV100-R5 -> 'rotation 5°'."""
    name = condition.replace('SV100-', '')
    if name == 'GT':
        return 'none (GT poses)'
    return ('translation %s%%' if name[0] == 'T' else 'rotation %s°') % name[1:]


def style(plt):
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE, 'legend.frameon': False})


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def psnr_table():
    """Full-test-set PSNR for every (condition, method) from the archived summaries."""
    table = {}
    for row in read_csv(ARTIFACTS/'pose_optimization/summary.csv'):
        table[row['case'], row['method']] = float(row['psnr'])
    for row in read_csv(ARTIFACTS/'warmup/summary.csv'):
        if row['method'] != 'fixed':
            table[row['case'], row['method']+'_warmup'] = float(row['psnr'])
    return table


def psnr_matrix(plt, output):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    table = psnr_table()
    values = np.array([[table[c, m] for m in METHODS] for c in CONDITIONS])
    cmap = LinearSegmentedColormap.from_list('blue', SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), layout='constrained')
    ax.imshow(values, cmap=cmap, vmin=11, vmax=28, aspect='auto')
    for i, row in enumerate(values):
        best = row.argmax()
        for j, value in enumerate(row):
            dark = (value-11)/17 > .55
            ax.text(j, i, '%.1f' % value, ha='center', va='center', fontsize=10,
                    color='#ffffff' if dark else INK, fontweight='bold' if j == best else 'normal')
    ax.set_xticks(range(5), [LABELS[m].replace(' + ', '\n+ ') for m in METHODS])
    ax.set_yticks(range(9), [short(c) for c in CONDITIONS])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 9), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2); ax.tick_params(which='minor', length=0)
    ax.axhline(.5, color=SURFACE, linewidth=5); ax.axhline(4.5, color=SURFACE, linewidth=5)
    ax.set_ylabel('Training-pose perturbation')
    ax.set_title('Test PSNR (dB, 200 views, 800×800) — bold = best per row', loc='left', fontsize=11)
    fig.savefig(output/'psnr_matrix.png', dpi=160); plt.close(fig)


def drift_over_training(plt, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained', sharey=True)
    for ax, method, color in zip(axes, ('naive', 'camp'), (BLUE, ORANGE)):
        path = ARTIFACTS/('training_curves/diagnostics/%s_SV100-GT.jsonl' % method)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps = [r['step'] for r in rows]
        raw = [r['pose_errors']['raw']['translation_radius_pct_mean'] for r in rows]
        aligned = [r['pose_errors']['aligned']['translation_radius_pct_mean'] for r in rows]
        ax.fill_between(steps, aligned, raw, color=color, alpha=.12, linewidth=0)
        ax.plot(steps, raw, color=color, linewidth=2)
        ax.plot(steps, aligned, color=color, linewidth=2, linestyle=(0, (4, 2)))
        ax.text(steps[-1], raw[-1], ' raw  %.2f%%' % raw[-1], va='center', fontsize=9, color=INK)
        ax.text(steps[-1], aligned[-1], ' Sim(3)-aligned  %.2f%%' % aligned[-1], va='center', fontsize=9, color=INK)
        ax.text(steps[-6], (raw[-6]*aligned[-6])**.5, 'global drift', ha='center', va='center', fontsize=9, color=MUTED)
        ax.set_yscale('log'); ax.set_xlim(0, 41000); ax.set_xticks(range(0, 30001, 10000))
        ax.set_title('%s, initialized at ground-truth poses' % LABELS[method], loc='left', fontsize=10)
        ax.set_xlabel('Training step'); ax.grid(axis='y', color=GRID, linewidth=.8)
    axes[0].set_ylabel('Camera-center error\n(% of mean camera radius, log)')
    fig.savefig(output/'drift_over_training.png', dpi=160); plt.close(fig)


def square_crop(image, margin=36):
    """Square crop around the non-white object, so every tile in a figure shares one window."""
    import numpy as np
    pixels = np.asarray(image.convert('RGB')).astype(int)
    ys, xs = np.where((pixels < 235).any(-1))
    side = max(xs.max()-xs.min(), ys.max()-ys.min()) + 2*margin
    cx, cy = (xs.min()+xs.max())//2, (ys.min()+ys.max())//2
    left = int(min(max(cx-side//2, 0), image.width-side)); top = int(min(max(cy-side//2, 0), image.height-side))
    return (left, top, left+side, top+side)


def render_metrics():
    rows = read_csv(ARTIFACTS/'showcase_renders/metrics.csv')
    return {(r['condition'], r['method'], r['mode'], int(r['view'])): float(r['psnr']) for r in rows}


def hero(plt, stills, output, view=100):
    from PIL import Image
    from matplotlib import cm, colors
    metrics = render_metrics()
    gt = Image.open(stills/('SV100-GT/gt/view_%03d.png' % view))
    box = square_crop(gt)
    tiles = [('Ground truth', gt, None, None),
             ('Fixed cameras', 'fixed/view_%03d_raw', 'fixed', 'raw'),
             ('Naive pose opt. (raw)', 'naive/view_%03d_raw', 'naive', 'raw'),
             ('Naive + Sim(3) alignment', 'naive/view_%03d_aligned', 'naive', 'aligned')]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), layout='constrained')
    for col, (title, name, method, mode) in enumerate(tiles):
        top, bottom = axes[0, col], axes[1, col]
        if method is None:
            top.imshow(gt.crop(box)); top.set_title(title, fontsize=11)
            bottom.axis('off')
            bottom.text(.02, .5, 'Same checkpoint, same test camera.\nThe raw render is sharp but shifted:\n'
                        'error concentrates on edges.\nA Sim(3) fitted on training camera\ncenters only '
                        'removes most of it.', transform=bottom.transAxes, va='center', fontsize=9.5, color=MUTED)
        else:
            path = stills/'SV100-GT'/(name % view)
            top.imshow(Image.open(str(path)+'.png').crop(box))
            top.set_title('%s\n%.2f dB' % (title, metrics['SV100-GT', method, mode, view]), fontsize=11)
            error = name.replace('_raw', '_err_raw').replace('_aligned', '_err_aligned')
            bottom.imshow(Image.open(str(stills/'SV100-GT'/(error % view))+'.png').crop(box))
        for ax in (top, bottom):
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
    bar = fig.colorbar(cm.ScalarMappable(colors.Normalize(0, .3), 'turbo'), ax=axes[1, 1:], shrink=.8, pad=.01)
    bar.set_label('per-pixel RGB error ‖pred − gt‖₂', fontsize=9); bar.outline.set_visible(False)
    fig.savefig(output/'hero_drift.png', dpi=110); plt.close(fig)


def perturbation_grid(plt, stills, output, view=175):
    from PIL import Image
    metrics = render_metrics()
    gt = Image.open(stills/('SV100-T5/gt/view_%03d.png' % view))
    box = square_crop(gt)
    rows = [('SV100-T5', 'Translation noise 5%'), ('SV100-R5', 'Rotation noise 5°')]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6), layout='constrained')
    for r, (condition, name) in enumerate(rows):
        axes[r, 0].imshow(gt.crop(box)); axes[r, 0].set_title('Ground truth', fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=11)
        for c, method in enumerate(METHODS, 1):
            image = Image.open(stills/condition/method/('view_%03d_raw.png' % view))
            axes[r, c].imshow(image.crop(box))
            axes[r, c].set_title('%s\n%.1f dB' % (LABELS[method], metrics[condition, method, 'raw', view]), fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.savefig(output/'perturbation_grid.png', dpi=100); plt.close(fig)


def perturbed_poses(gt_transforms, condition_args):
    """Re-create a perturbed training split with the original script and seed."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)/'lego'
        source.mkdir()
        shutil.copy2(gt_transforms, source/'transforms_train.json')
        subprocess.run([sys.executable, str(ROOT/'scripts/data/perturb_poses.py'), '--src', str(source),
                        '--dst', str(Path(directory)/'noisy'), *condition_args, '--seed', '42'],
                       check=True, capture_output=True)
        frames = json.loads((Path(directory)/'noisy/transforms_train.json').read_text())['frames']
    return [frame['transform_matrix'] for frame in frames]


def camera_correction(plt, gt_transforms, output, magnify=10):
    """T5 training-camera centers: injected noise vs. what Naive + warm-up leaves after optimization."""
    import numpy as np
    from poseguard.alignment import fit_sim3, transform_poses
    gt = np.array([f['transform_matrix'] for f in json.loads(Path(gt_transforms).read_text())['frames']])[:, :3]
    noisy = np.array(perturbed_poses(gt_transforms, ['--translation-ratio', '0.05']))[:, :3]
    record = json.loads((ROOT/'artifacts/experiments/warmup/naive/SV100-T5/optimized_poses.json').read_text())
    optimized = np.array(record['poses'])[:, :3]
    stored = record['initial_errors'].get('raw', record['initial_errors'])['translation_radius_pct_mean']
    radius = np.linalg.norm(gt[:, :, 3], axis=1).mean()
    error = lambda poses: np.linalg.norm(poses[:, :, 3]-gt[:, :, 3], axis=1).mean()
    if abs(100*error(noisy)/radius-stored) > 1e-6:
        raise ValueError('Regenerated perturbation does not match the archived run')
    aligned = transform_poses(optimized, fit_sim3(optimized[:, :, 3], gt[:, :, 3]))
    panels = [(noisy, '#52514e', 'Injected noise (5%% translation)\nmean camera-center error %.3f' % error(noisy)),
              (optimized, BLUE, 'After Naive + warm-up, 30k steps\nmean error %.3f, %.3f after one global Sim(3)'
               % (error(optimized), error(aligned)))]
    fig = plt.figure(figsize=(11, 4.4))
    for index, (poses, color, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        centers, offsets = gt[:, :, 3], (poses[:, :, 3]-gt[:, :, 3])*magnify
        for start, offset in zip(centers, offsets):
            ax.plot(*np.stack((start, start+offset)).T, color=color, linewidth=1.6, solid_capstyle='round')
        ax.scatter(*centers.T, s=12, color='#b5b4ae', depthshade=False)
        ax.scatter([0], [0], [0], s=60, marker='s', color=INK)
        ax.text(0, 0, -.7, 'scene', ha='center', fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, loc='left', y=1.0, pad=0)
        ax.view_init(elev=24, azim=-60)
        ax.set_box_aspect((1, 1, .55), zoom=1.2)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-.8, 5.2)
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=-.08, top=.86, wspace=0)
    fig.text(.5, .015, 'Lego training cameras (gray, %.2f units from the scene). Segments: camera-center error, '
             'drawn ×%d.' % (radius, magnify), ha='center', fontsize=9.5, color=MUTED)
    fig.savefig(output/'camera_correction.png', dpi=130); plt.close(fig)


def side_by_side_gif(delivery, output):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        print('ffmpeg not found; skipping GIF'); return
    graph = ('fps=12,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];'
             '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle')
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(delivery/'videos/side_by_side_T5.mp4'),
                    '-filter_complex', graph, '-loop', '0', str(output/'t5_side_by_side.gif')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, help='extracted delivery_<date>/ directory (renders, videos)')
    parser.add_argument('--gt-transforms', type=Path, help='NeRF-Synthetic lego/transforms_train.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    args.output.mkdir(parents=True, exist_ok=True)
    psnr_matrix(plt, args.output)
    drift_over_training(plt, args.output)
    if args.delivery:
        hero(plt, args.delivery/'stills', args.output)
        perturbation_grid(plt, args.delivery/'stills', args.output)
        side_by_side_gif(args.delivery, args.output)
    if args.gt_transforms:
        camera_correction(plt, args.gt_transforms, args.output)
    for path in sorted(args.output.iterdir()):
        print('%-28s %7.0f KB' % (path.name, path.stat().st_size/1024))


if __name__ == '__main__':
    main()
