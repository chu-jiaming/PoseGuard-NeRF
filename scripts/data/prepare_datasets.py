"""Create the eight seed-42 pose perturbations without overwriting datasets."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training.sv100_pipeline import CASES, DATA, GT


def main():
    if not (GT / 'transforms_train.json').is_file():
        raise SystemExit('Missing Lego dataset: ' + str(GT))
    targets = [(kind, magnitude, DATA / directory)
               for _, kind, magnitude, directory in CASES if kind != 'gt']
    existing = [str(path) for _, _, path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise SystemExit('Refusing to overwrite existing datasets: ' + ', '.join(existing))
    for kind, magnitude, path in targets:
        flag = '--translation-ratio' if kind == 'translation' else '--rotation-deg'
        subprocess.run([sys.executable, str(ROOT / 'scripts/data/perturb_poses.py'),
                        '--src', str(GT), '--dst', str(path), flag, str(magnitude),
                        '--seed', '42'], check=True)


if __name__ == '__main__':
    main()
