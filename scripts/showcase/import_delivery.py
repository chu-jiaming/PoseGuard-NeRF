#!/usr/bin/env python
"""Copy the small, publishable part of a server showcase delivery into artifacts/.

The delivery directory is the extracted delivery_<date>/ produced on the training
server (renders, videos, TensorBoard scalars, pose diagnostics). Large binaries
stay out of Git: only per-image metrics, slimmed training curves, diagnostics and
provenance records are copied. README figures are built by make_figures.py.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
# Eval tags are kept in full; dense per-step tags are thinned (see keep()).
EVAL_TAGS = {'Eval Images Metrics/fine_psnr', 'Eval Images Metrics Dict (all images)/fine_psnr',
             'Eval Images Metrics Dict (all images)/fine_ssim', 'Eval Images Metrics Dict (all images)/fine_lpips'}
DENSE_TAGS = {'Train Loss', 'learning_rate/camera_opt'}
RECORDS = ('README.md', 'inventory.json', 'checkpoints_index.json')


def keep(tag, step):
    if tag in EVAL_TAGS:
        return True
    # Dense near the 2,500-step warm-up ramp, sparse afterwards; always keep the final step.
    return tag in DENSE_TAGS and (step % (50 if step < 5000 else 500) == 0 or step == 29999)


def import_curves(delivery, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (delivery/'curves/tb_scalars.csv').open() as source, (output/'tb_scalars.csv').open('w', newline='') as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if keep(row['tag'], int(row['step'])):
                writer.writerow(row); rows += 1
    diagnostics = output/'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for path in sorted((delivery/'curves/diagnostics').glob('*.jsonl')):
        shutil.copy2(path, diagnostics/path.name)
    return rows


def import_render_metrics(delivery, output):
    manifest = json.loads((delivery/'manifest.json').read_text())
    fields = ['condition', 'method', 'mode', 'view', 'psnr', 'ssim', 'lpips',
              'near_effective', 'far_effective', 'check_type', 'check_passed', 'file']
    rows = []
    for name, record in sorted(manifest['files'].items()):
        if record.get('kind') != 'render' or record.get('video'):
            continue
        check = record.get('check') or {}
        rows.append(dict(condition=record['condition'], method=record['method'], mode=record['mode'],
                         view=record['view'], **record['metrics'], near_effective=record['near_effective'],
                         far_effective=record['far_effective'], check_type=check.get('type'),
                         check_passed=check.get('passed'), file=name))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output/'error_map.json').write_text(json.dumps(manifest['error_map'], indent=2)+'\n')
    return len(rows)


def import_records(delivery, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in RECORDS:
        target = output/('DELIVERY_REPORT.md' if name == 'README.md' else name)
        shutil.copy2(delivery/name, target)
    shutil.copytree(delivery/'scripts', output/'server_scripts', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('delivery', type=Path, help='extracted delivery_<date>/ directory')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    args = parser.parse_args()
    if not (args.delivery/'manifest.json').is_file():
        parser.error('Not a delivery directory (manifest.json missing): %s' % args.delivery)
    curves = import_curves(args.delivery, args.artifacts/'analysis/training_curves')
    renders = import_render_metrics(args.delivery, args.artifacts/'analysis/showcase_renders')
    import_records(args.delivery, args.artifacts/'records/showcase')
    print('curve rows: %d; still metrics: %d' % (curves, renders))


if __name__ == '__main__':
    main()
