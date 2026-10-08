"""Exercise durability without spending GPU time or touching real datasets."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import pose_optimization_pipeline as pipeline


class OrchestrationTests(unittest.TestCase):
    def test_failure_continues_and_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            observed = []
            def execute(args, case, method, entry, persist):
                observed.append(method+'/'+case[0])
                if method == 'naive':
                    raise RuntimeError('simulated CUDA failure')
                entry.update(status='SUCCESS', stage='complete'); persist()
            def report(args, state, baselines, label):
                (args.run_dir/'report.md').write_text(label)
            arguments = ['pipeline', '--run-dir', str(directory), '--cases', 'SV100-GT']
            with patch.object(pipeline, 'preflight', return_value=({}, {'data': {'SV100-GT': {}}})), \
                 patch.object(pipeline, 'validate', return_value={}), \
                 patch.object(pipeline, 'run'), patch.object(pipeline, 'report', side_effect=report), \
                 patch.object(pipeline, 'comparisons'), patch.object(pipeline, 'execute_case', side_effect=execute), \
                 patch('sys.argv', arguments), patch('sys.stdout'), patch('sys.stderr'):
                self.assertEqual(pipeline.main(), 1)
            self.assertEqual(observed, ['naive/SV100-GT', 'camp/SV100-GT'])
            state = json.loads((directory/'state.json').read_text())
            self.assertEqual(state['experiments']['naive/SV100-GT']['status'], 'FAILED')
            self.assertEqual(state['experiments']['camp/SV100-GT']['status'], 'SUCCESS')
            self.assertTrue((directory/'failure_naive_SV100-GT.txt').is_file())
            observed.clear()
            # Verified success short-circuit is covered by checkpoint integration smoke;
            # here remove its fake success so the remaining independent case can be observed.
            state['experiments']['camp/SV100-GT']['status'] = 'PENDING'
            state['error'] = 'old interruption'
            (directory/'state.json').write_text(json.dumps(state))
            with patch.object(pipeline, 'preflight', return_value=({}, {'data': {'SV100-GT': {}}})), \
                 patch.object(pipeline, 'validate', return_value={}), patch.object(pipeline, 'run'), \
                 patch.object(pipeline, 'report', side_effect=report), patch.object(pipeline, 'comparisons'), \
                 patch.object(pipeline, 'execute_case', side_effect=execute), \
                 patch('sys.argv', arguments+['--resume']), patch('sys.stdout'), patch('sys.stderr'):
                self.assertEqual(pipeline.main(), 1)
            self.assertEqual(observed, ['camp/SV100-GT'])
            self.assertNotIn('error', json.loads((directory/'state.json').read_text()))
            observed.clear()
            def succeed(args, case, method, entry, persist):
                observed.append(method+'/'+case[0]); entry.update(status='SUCCESS'); persist()
            state = json.loads((directory/'state.json').read_text())
            state['experiments']['camp/SV100-GT']['status'] = 'PENDING'
            (directory/'state.json').write_text(json.dumps(state))
            with patch.object(pipeline, 'preflight', return_value=({}, {'data': {'SV100-GT': {}}})), \
                 patch.object(pipeline, 'validate', return_value={}), patch.object(pipeline, 'run'), \
                 patch.object(pipeline, 'report', side_effect=report), patch.object(pipeline, 'comparisons'), \
                 patch.object(pipeline, 'execute_case', side_effect=succeed), \
                 patch('sys.argv', arguments+['--resume', '--retry-failed']), patch('sys.stdout'), patch('sys.stderr'):
                self.assertEqual(pipeline.main(), 0)
            self.assertEqual(observed, ['naive/SV100-GT', 'camp/SV100-GT'])

    def test_interrupted_run_and_manifest_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            arguments = ['pipeline', '--run-dir', str(directory), '--cases', 'SV100-GT']
            with patch.object(pipeline, 'preflight', return_value=({}, {'data': {'SV100-GT': {}}})), \
                 patch.object(pipeline, 'validate', return_value={}), patch.object(pipeline, 'run'), \
                 patch.object(pipeline, 'execute_case', side_effect=KeyboardInterrupt()), \
                 patch('sys.argv', arguments), patch('sys.stdout'), patch('sys.stderr'):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.main()
            state = json.loads((directory/'state.json').read_text())
            self.assertEqual(state['status'], 'INTERRUPTED')
            self.assertEqual(state['experiments']['naive/SV100-GT']['status'], 'INTERRUPTED')
            with patch.object(pipeline, 'preflight', return_value=({}, {'data': {'SV100-GT': {'modified': True}}})), \
                 patch('sys.argv', arguments+['--resume']):
                with self.assertRaisesRegex(RuntimeError, 'fingerprints changed'):
                    pipeline.main()

    def test_resume_searches_older_checkpoint_after_empty_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(steps=10, interval=2, run_dir=root, smoke=True)
            entry = {'attempts': [str(root/'attempt_001'), str(root/'attempt_002')]}
            calls = []
            def latest(attempt, target):
                if attempt.name == 'attempt_001':
                    return attempt/'step-000000003.ckpt', 3
                return None, -1
            def command_run(command, logfile):
                calls.append(command)
                raise RuntimeError('stop after observing resumed command')
            with patch.object(pipeline, 'latest_checkpoint', side_effect=latest), \
                 patch.object(pipeline, 'run', side_effect=command_run):
                with self.assertRaisesRegex(RuntimeError, 'stop after observing'):
                    pipeline.execute_case(args, pipeline.CASES[0], 'naive', entry, lambda: None)
            self.assertIn('--checkpoint', calls[0])
            self.assertEqual(calls[0][calls[0].index('--checkpoint')+1], root/'attempt_001/step-000000003.ckpt')
            self.assertEqual(len(entry['attempts']), 3)


if __name__ == '__main__':
    unittest.main()
