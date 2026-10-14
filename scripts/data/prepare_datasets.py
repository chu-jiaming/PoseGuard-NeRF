"""Create the eight seed-42 pose perturbations without overwriting datasets."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training.sv100_pipeline import CASES, DATA
from scripts.runtime import PATH_HELP


def main():
    parser=argparse.ArgumentParser(description=__doc__, epilog=PATH_HELP)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args=parser.parse_args()
    data=args.data_root.expanduser().resolve()
    GT=data/'lego'
    if not (GT / 'transforms_train.json').is_file():
        raise SystemExit('Missing Lego dataset: ' + str(GT))
    targets = [(kind, magnitude, data / directory)
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
