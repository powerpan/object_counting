from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_round2_s_first as flow
from run_round2_validation_workflow import read, write, sha


class SFirstTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.result=self.root/'results/round2-v1'
        self.result.mkdir(parents=True)
        self.cfg={'total_M4_seconds':108000,'single_S_seconds':21600,
                  'single_L_seconds':43200,'single_validation_seconds':5400}
        self.previous={'state':'running','stage':'head-S-seed42','spent_seconds':42.,
                       'completed':['cpu-tests','gate','capacity-S','capacity-L']}

    def ledger(self):
        return flow.create_ledger(self.result/'ledger.json',self.previous,'original-sha',self.cfg)

    def settle(self,ledger,stage,elapsed=1.):
        ledger.begin(stage,21600,'start')
        self.assertTrue(ledger.finish(stage,elapsed,True,{},'end'))

    def test_partial_S_is_not_imported_and_completed_S_cannot_be_counted_twice(self):
        ledger=self.ledger();self.assertEqual(ledger.spent,42.)
        self.settle(ledger,'head-S-seed42',16000.)
        self.assertEqual(ledger.spent,16042.)
        with self.assertRaises(FileExistsError):self.ledger()
        bad={**self.previous,'completed':self.previous['completed']+['head-S-seed42']}
        with self.assertRaises(ValueError):
            flow.create_ledger(self.result/'duplicate.json',bad,'sha',self.cfg)
        self.assertFalse((self.result/'duplicate.json').exists())

    def test_L_requires_both_audits_and_accounting_even_if_files_claim_success(self):
        ledger=self.ledger()
        for label,n in [('gate-density-head-S-seed42',32),('val-head-S-seed42',1286)]:
            (self.result/label).mkdir();write(self.result/label/'audit.json',{'status':'passed','n':n})
            self.settle(ledger,label)
        with self.assertRaises(ValueError):flow.require_s_validation(self.root,ledger)
        self.settle(ledger,'audit-gate-density-head-S-seed42')
        with self.assertRaises(ValueError):flow.require_s_validation(self.root,ledger)
        self.settle(ledger,'audit-val-head-S-seed42')
        flow.require_s_validation(self.root,ledger)
        write(self.result/'val-head-S-seed42/audit.json',{'status':'passed','n':1285})
        with self.assertRaises(ValueError):flow.require_s_validation(self.root,ledger)

    def simulated_sequence(self, fail_at=None):
        ledger=self.ledger();state={'completed':[]};state_file=self.result/'state.json';seen=[]
        def runner(root,ledger,state,state_file,stage,cap,command,verify):
            seen.append(stage);limit=ledger.begin(stage,cap,'start');command(limit-10)
            if stage==fail_at:
                ledger.finish(stage,1.,False,{},'end')
                raise RuntimeError('Injected audit failure')
            if stage.startswith('training-audit-'):
                width=stage.split('-')[-1][0];d=self.result/('head-'+width+'-seed42');d.mkdir(exist_ok=True)
                write(d/'training_audit_s_first_v1.json',{'status':'passed'})
            elif stage=='head-L-seed42':
                d=self.result/stage;d.mkdir()
                write(d/'status.json',{'status':'complete'});write(d/'checkpoint_manifest.json',{'complete_epochs':30})
            elif stage.startswith('audit-'):
                label=stage[len('audit-'):];n=32 if label.startswith('gate-') else 1286
                write(self.result/label/'audit.json',{'status':'passed','n':n})
            else:
                d=self.result/stage;d.mkdir();n=32 if stage.startswith('gate-') else 1286
                # Very poor scores must not silently cancel or retune L.
                write(d/'summary.json',{'status':'complete','n':n,'overall':{'n':n,'mae':999.}})
                write(d/'protocol.json',{'synthetic':True})
            evidence=verify();ledger.finish(stage,1.,True,evidence,'end');state['completed'].append(stage)
        with mock.patch.object(flow,'run_stage',side_effect=runner):
            if fail_at:
                with self.assertRaises(RuntimeError):flow.continue_after_s(self.root,ledger,state,state_file,self.cfg)
            else:
                flow.continue_after_s(self.root,ledger,state,state_file,self.cfg)
        return seen,ledger,state

    def test_full_order_and_poor_S_scores_do_not_change_L_training(self):
        seen,ledger,state=self.simulated_sequence()
        self.assertLess(seen.index('audit-gate-density-head-S-seed42'),seen.index('val-head-S-seed42'))
        self.assertLess(seen.index('audit-val-head-S-seed42'),seen.index('head-L-seed42'))
        self.assertLess(seen.index('audit-val-head-L-seed42'),seen.index('m4-crossfit-seed42'))
        self.assertEqual(seen[-1],'audit-m4-crossfit-seed42')
        self.assertEqual(state['state'],'awaiting_seed_sensitivity')
        self.assertFalse(state['all_round2_complete'])
        context=read(self.result/'L_delegated_budget_s_first_v1.json')
        self.assertEqual(context['delegated_training_seconds'],43200-10)
        self.assertGreater(context['global_ledger_remaining_seconds'],context['delegated_training_seconds'])

    def test_S_audit_failure_stops_before_L_and_retains_its_cost(self):
        seen,ledger,state=self.simulated_sequence('audit-val-head-S-seed42')
        self.assertNotIn('head-L-seed42',seen)
        self.assertEqual(ledger.data['state'],'stopped')
        self.assertGreater(ledger.spent,42.)
        self.assertFalse((self.result/'s_validation_ready_v1.json').exists())

    def test_process_identity_prevents_signalling_reused_PID(self):
        with mock.patch.object(flow,'process',return_value={'start_ticks':2,'state':'T'}), mock.patch.object(flow.os,'kill') as kill:
            with self.assertRaises(RuntimeError):flow.stop_identity(111,1)
            kill.assert_not_called()
        with mock.patch.object(flow,'process',side_effect=[{'start_ticks':1,'state':'T'},None]), mock.patch.object(flow.os,'kill') as kill:
            flow.stop_identity(111,1)
            self.assertEqual(kill.call_args_list,[mock.call(111,signal.SIGTERM),mock.call(111,signal.SIGCONT)])

    def test_S_training_audit_does_not_require_L_but_rejects_changed_frozen_state(self):
        d=self.result/'head-S-seed42';d.mkdir();work=self.root/'work/round2-v1/head-S-seed42';work.mkdir(parents=True)
        for kind in ('best','last'):(work/(kind+'.pth')).write_bytes(b'synthetic')
        history=[{'epoch':i,'train_n':3457,'holdout_n':202,'train_loss':1.,'holdout_loss':1.,
                  'holdout_mae':1.,'elapsed_seconds':i*10.,'frozen_state_sha256':'fixed'} for i in range(1,31)]
        write(d/'history.json',history)
        write(d/'status.json',{'status':'complete','epoch':30,'best_epoch':1,'best_holdout_mae':1.,'elapsed_seconds':301.})
        write(d/'checkpoint_manifest.json',{'complete_epochs':30,'best_epoch':1,'best_holdout_mae':1.,
              'frozen_model_sha256':'fixed',**{k+'_sha256':sha(work/(k+'.pth')) for k in ('best','last')}})
        self.assertEqual(flow.check_one_training(self.root,'S')['status'],'passed')
        history[-1]['frozen_state_sha256']='changed';write(d/'history.json',history)
        with self.assertRaises(ValueError):flow.check_one_training(self.root,'S')


if __name__=='__main__':unittest.main()
