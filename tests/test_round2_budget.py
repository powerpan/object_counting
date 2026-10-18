import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.round2.budget import BudgetLedger
from run_round2_validation_workflow import check_training, exclusive, read, run_stage, sha, write


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'logs').mkdir()
        self.source = {'state': 'awaiting_validation', 'spent_seconds': 40.,
                       'completed': ['head-S-seed42', 'head-L-seed42']}

    def ledger(self, cap=100.):
        return BudgetLedger.create(self.root/'ledger.json', cap, self.source, 'source-sha')

    def test_handoff_already_includes_both_heads_and_cannot_be_imported_twice(self):
        ledger = self.ledger()
        self.assertEqual(ledger.spent, 40.)
        ledger.begin('density', 50., 'begin')
        self.assertTrue(ledger.finish('density', 5., True, {'result': 'sha'}, 'end'))
        reopened = BudgetLedger(ledger.path)
        self.assertEqual(reopened.spent, 45.)
        self.assertEqual(reopened.remaining, 55.)
        with self.assertRaises(FileExistsError):
            self.ledger()
        with self.assertRaises(FileExistsError):
            reopened.begin('density', 50., 'duplicate')

    def test_incomplete_training_or_unresolved_child_does_not_create_a_ledger(self):
        for change in ({'state': 'running'}, {'completed': ['head-S-seed42']}, {'stage_pid': 123}):
            with self.assertRaises(ValueError):
                BudgetLedger.create(self.root/'ledger.json', 100, {**self.source, **change}, 'sha')
            self.assertFalse((self.root/'ledger.json').exists())

    def test_interruption_cannot_reset_reserved_budget_by_reopening(self):
        ledger = self.ledger()
        ledger.begin('interrupted', 50., 'begin')
        reopened = BudgetLedger(ledger.path)
        with self.assertRaises(RuntimeError):
            reopened.begin('replacement', 50., 'later')
        self.assertEqual(reopened.data['active']['id'], 'interrupted')

    def test_failed_work_is_charged_and_blocks_dependent_jobs(self):
        ledger = self.ledger()
        ledger.begin('failed', 50., 'begin')
        self.assertFalse(ledger.finish('failed', 12., False, {}, 'end'))
        self.assertEqual(ledger.spent, 52.)
        with self.assertRaises(RuntimeError):
            BudgetLedger(ledger.path).begin('next', 50., 'later')

    def test_stage_cap_tightens_to_total_and_overrun_is_never_accepted(self):
        ledger = self.ledger()
        self.assertEqual(ledger.begin('last', 1000., 'begin'), 60.)
        self.assertFalse(ledger.finish('last', 61., True, {}, 'end'))
        self.assertEqual(ledger.spent, 101.)
        self.assertEqual(ledger.remaining, 0.)
        self.assertEqual(ledger.data['state'], 'stopped')

    def test_nonfinite_negative_and_invalid_elapsed_values_are_rejected(self):
        for bad in (float('nan'), float('inf'), -1, True, '5'):
            with self.assertRaises(ValueError):
                BudgetLedger.create(self.root/'ledger.json', 100,
                                    {**self.source, 'spent_seconds': bad}, 'sha')
        ledger = self.ledger()
        ledger.begin('stage', 50., 'begin')
        with self.assertRaises(ValueError):
            ledger.finish('stage', float('nan'), True, {}, 'end')
        self.assertIsNotNone(BudgetLedger(ledger.path).data['active'])

    def test_existing_gpu_owner_prevents_a_second_controller(self):
        path = self.root/'gpu.lock'
        code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
                'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
        with exclusive(path):
            result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'BlockingIOError', result.stderr)
        self.assertEqual(subprocess.run([sys.executable, '-c', code, str(path)],
                                        capture_output=True).returncode, 0)

    def run_command(self, code, cap=50., verify=lambda: {}):
        ledger = self.ledger()
        state = {'completed': []}
        path = self.root/'state.json'
        run_stage(self.root, ledger, state, path, 'child', cap,
                  lambda limit: [sys.executable, '-c', code], verify)
        return ledger, read(path)

    def test_successful_exit_requires_independent_result_verification(self):
        def reject():
            raise ValueError('No completed audit')
        with self.assertRaises(ValueError):
            self.run_command('pass', verify=reject)
        ledger = BudgetLedger(self.root/'ledger.json')
        self.assertEqual(ledger.data['state'], 'stopped')
        self.assertGreater(ledger.spent, 40.)
        self.assertEqual(read(self.root/'state.json')['completed'], [])

    def test_nonzero_child_is_settled_without_accepting_its_partial_output(self):
        with self.assertRaises(RuntimeError):
            self.run_command('raise SystemExit(2)')
        ledger = BudgetLedger(self.root/'ledger.json')
        self.assertEqual(ledger.data['entries'][-1]['outcome'], 'stopped')
        self.assertIsNone(ledger.data['active'])
        self.assertNotIn('stage_pid', read(self.root/'state.json'))

    def test_timeout_stops_child_and_retains_failed_time(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_command('import time; time.sleep(60)', cap=11.)
        ledger = BudgetLedger(self.root/'ledger.json')
        self.assertEqual(ledger.data['state'], 'stopped')
        self.assertGreaterEqual(ledger.spent, 40.5)
        self.assertIsNone(ledger.data['active'])
        self.assertEqual(read(self.root/'state.json')['completed'], [])


class TrainingAuditTests(unittest.TestCase):
    def test_tampered_checkpoint_and_wrong_holdout_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = 'unchanged'
            cfg = {'m4': {'epochs': 30}}
            amendment = {'m4_S_single_run_seconds': 21600, 'm4_L_single_run_seconds': 43200}
            for width in ('S','L'):
                label = 'head-'+width+'-seed42'
                out = root/'results/round2-v1'/label
                weights = root/'work/round2-v1'/label
                out.mkdir(parents=True); weights.mkdir(parents=True)
                history = [{'epoch': i, 'train_n': 3457, 'holdout_n': 202,
                            'train_loss': 1., 'holdout_loss': 2., 'holdout_mae': 5. if i in (4,5) else 6.,
                            'elapsed_seconds': i*10., 'frozen_state_sha256': frozen} for i in range(1,31)]
                for kind in ('best','last'):
                    (weights/(kind+'.pth')).write_bytes(b'synthetic-checkpoint')
                selection = {'best_epoch': 4, 'best_holdout_mae': 5.}
                write(out/'history.json', history)
                write(out/'status.json', {'status': 'complete', 'epoch': 30,
                                          'elapsed_seconds': 301., **selection})
                write(out/'checkpoint_manifest.json', {'complete_epochs': 30, **selection,
                      'frozen_model_sha256': frozen,
                      **{k+'_sha256': sha(weights/(k+'.pth')) for k in ('best','last')}})
            write(root/'results/round2-v1/workflow_before_budget_amendment.json',
                  {'state': 'running', 'stage': 'head-S-seed42', 'spent_seconds': 42.})
            check_training(root, cfg, amendment, {'spent_seconds': 650.})
            with self.assertRaisesRegex(ValueError, 'undercounts'):
                check_training(root, cfg, amendment, {'spent_seconds': 603.})
            path = root/'results/round2-v1/head-S-seed42/status.json'
            status = read(path)
            write(path, {**status, 'best_epoch': 5})
            with self.assertRaisesRegex(ValueError, 'selection'):
                check_training(root, cfg, amendment, {'spent_seconds': 650.})
            write(path, status)
            (root/'work/round2-v1/head-L-seed42/best.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint content'):
                check_training(root, cfg, amendment, {'spent_seconds': 650.})


if __name__ == '__main__':
    unittest.main()
