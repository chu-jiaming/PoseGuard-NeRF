"""Check published entrypoints outside the checkout without GPU training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptPortabilityTests(unittest.TestCase):
    def test_all_python_help_without_datasets(self):
        with tempfile.TemporaryDirectory(prefix='poseguard external ') as directory:
            env = dict(os.environ, POSEGUARD_DATA=str(Path(directory)/'missing data'),
                       POSEGUARD_RUN_ROOT=str(Path(directory)/'missing runs'))
            for path in ROOT.joinpath('scripts').glob('*/*.py'):
                with self.subTest(script=path.name):
                    result = subprocess.run([sys.executable, str(path), '--help'],
                                            cwd=directory, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout.lower())
            self.assertFalse((Path(directory)/'missing runs').exists())

    def test_relative_environment_paths_are_checkout_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT), POSEGUARD_DATA='custom data',
                       POSEGUARD_RUN_ROOT='custom runs')
            for key in ('POSEGUARD_BASELINE_SUMMARY', 'POSEGUARD_ORIGINAL_RUN', 'POSEGUARD_WARMUP_RUN'):
                env.pop(key, None)
            result = subprocess.run([sys.executable, '-c',
                'import json; from scripts.runtime import DATA, RUN_ROOT, WARMUP_RUN; '
                'print(json.dumps(list(map(str, (DATA, RUN_ROOT, WARMUP_RUN)))))'],
                cwd=directory, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), [str(ROOT/'custom data'),
                             str(ROOT/'custom runs'), str(ROOT/'custom runs/results/poseopt_sv100_warmup2500')])

    def test_default_analysis_from_external_directory(self):
        with tempfile.TemporaryDirectory(prefix='poseguard analysis ') as directory:
            output=Path(directory)/'nested output'
            subprocess.run([sys.executable, str(ROOT/'scripts/analysis/analyze_warmup_results.py'),
                            '--output', str(output)], cwd=directory, check=True, capture_output=True)
            actual=json.loads((output/'aggregate_metrics.json').read_text())
            expected=json.loads((ROOT/'artifacts/analysis/warmup/aggregate_metrics.json').read_text())
            self.assertEqual(actual, expected)

    def test_missing_audit_inputs_give_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable, str(ROOT/'scripts/evaluation/audit_pose_alignment.py'),
                                   '--gt', str(Path(directory)/'missing.json')], cwd=directory,
                                  capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Required file not found', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_audit_uses_explicit_run_and_output_directories(self):
        with tempfile.TemporaryDirectory(prefix='poseguard audit ') as directory:
            base=Path(directory)
            poses=[[[1,0,0,x],[0,1,0,y],[0,0,1,z]]
                   for x,y,z in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
            gt=base/'gt.json'
            gt.write_text(json.dumps({'frames':[{'transform_matrix':p} for p in poses]}))
            for name in ('original', 'warmup'):
                run=base/name; run.mkdir()
                (run/'config.yml').write_text('      scale_factor: 1.0\n')
                (run/'optimized_poses.json').write_text(json.dumps({'poses':poses}))
                entry={'status':'SUCCESS','config':str(run/'config.yml'),
                       'attempt':str(run),'metrics':{'psnr':20}}
                (run/'state.json').write_text(json.dumps({'experiments':{'naive/SV100-GT':entry}}))
            output=base/'nested'/'audit output'
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluation/audit_pose_alignment.py'),
                            '--gt',str(gt),'--original',str(base/'original'),
                            '--warmup',str(base/'warmup'),'--output',str(output)],
                           cwd=directory,check=True,capture_output=True)
            transforms=json.loads((output/'transforms.json').read_text())
            self.assertEqual(set(transforms), {'original/naive/SV100-GT','warmup/naive/SV100-GT'})
            for transform in transforms.values():
                self.assertAlmostEqual(transform['scale'],1.)
