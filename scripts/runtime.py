"""Portable runtime paths and preflight checks shared by command-line scripts.

Use POSEGUARD_DATA for the NeRF-Synthetic root and POSEGUARD_RUN_ROOT for
results/, outputs/ and logs/. Relative environment paths are repository-relative.
Existing runs can be selected with POSEGUARD_BASELINE_SUMMARY,
POSEGUARD_ORIGINAL_RUN and POSEGUARD_WARMUP_RUN. Saved checkpoints/configurations
must reference files available on the current machine; archives are not resumes.
"""
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default):
    path = Path(os.environ.get(name, str(default))).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


DATA = env_path('POSEGUARD_DATA', ROOT/'data/nerf_synthetic')
RUN_ROOT = env_path('POSEGUARD_RUN_ROOT', ROOT)
BASELINE_SUMMARY = env_path('POSEGUARD_BASELINE_SUMMARY', RUN_ROOT/'results/SV100_full_summary.csv')
ORIGINAL_RUN = env_path('POSEGUARD_ORIGINAL_RUN', RUN_ROOT/'results/poseopt_sv100')
WARMUP_RUN = env_path('POSEGUARD_WARMUP_RUN', RUN_ROOT/'results/poseopt_sv100_warmup2500')
PATH_HELP = (__doc__ or '').strip()


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit('Required file not found: %s. Select local data/runs with '
                         'POSEGUARD_DATA, POSEGUARD_BASELINE_SUMMARY, '
                         'POSEGUARD_ORIGINAL_RUN or POSEGUARD_WARMUP_RUN. '
                         'Historical server paths are not portable checkpoints.' % path)
    return path


def require_training():
    if not sys.platform.startswith('linux'):
        raise SystemExit('GPU training/rendering requires Linux and a CUDA-enabled PyTorch environment.')
    for package in ('torch', 'nerfstudio'):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit('Missing %s in %s; install the training dependencies with uv.' % (package, sys.executable))
        if package == 'nerfstudio' and version != '1.1.5':
            raise SystemExit('Requires Nerfstudio 1.1.5; found '+version)
    for command in ('ns-train', 'ns-eval', 'pgrep', 'uv'):
        if not shutil.which(command):
            raise SystemExit('Required command unavailable: '+command)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable in '+sys.executable)
