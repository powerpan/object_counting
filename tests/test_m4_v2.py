import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.m4v2 import ResidualHead,residual_loss,epoch_indices,expand_holdout,safety_stop,diagnostic_label,render_m1_grid
from run_m4_v2_workflow import import_budget


class M4V2Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_zero_residual_preserves_nonuniform_baseline_and_padding(self):
        h=ResidualHead();self.assertEqual(sum(p.numel() for p in h.parameters()),2144705)
        x=torch.randn(2,258,8,8);x[:,257]=torch.rand(2,8,8)
        mask=torch.zeros(2,1,8,8);mask[:,:,:5,:7]=1
        d=h(x,mask);self.assertTrue(torch.equal(d,x[:,257:258]*mask))
        self.assertEqual(float(d[:,:,5:,:].sum()),0.)
        d.sum().backward();self.assertTrue(torch.isfinite(h.final.weight.grad).all())
        self.assertGreater(float(h.final.weight.grad.abs().sum()),0.)

    def test_raw_count_and_negative_penalty_do_not_hide_cancellation(self):
        raw=torch.tensor([[[[1.,-1.]]]],requires_grad=True);target=torch.zeros_like(raw);mask=torch.ones_like(raw)
        loss,detail=residual_loss(raw,target,mask,torch.zeros(1))
        self.assertEqual(detail['raw_count'],0.);self.assertEqual(detail['clamped_count'],1.)
        self.assertEqual(detail['negative_mass'],1.);self.assertEqual(detail['negative_term'],1.)
        self.assertEqual(float(loss),10001.);loss.backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_padding_never_changes_loss_or_count(self):
        target=torch.zeros(1,1,2,2);target[0,0,0,0]=2
        mask=torch.zeros_like(target);mask[0,0,0,0]=1;raw=target.clone();raw[mask==0]=-1000
        loss,d=residual_loss(raw,target,mask,torch.tensor([2.]))
        self.assertEqual(float(loss),0.);self.assertEqual(d['negative_mass'],0.)

    def test_relative_error_cost_increases_with_count(self):
        t=torch.zeros(1,1,1,1);m=torch.ones_like(t)
        _,a=residual_loss(t,t,m,torch.tensor([100.]));_,b=residual_loss(t,t,m,torch.tensor([10000.]))
        self.assertAlmostEqual(b['count_term']/a['count_term'],10.,places=5)
        with self.assertRaises(FloatingPointError):residual_loss(t+float('nan'),t,m,torch.ones(1))

    def test_m1_mass_survives_boundary_and_dense_float32_conversion(self):
        boxes=np.array([[0,0,1/512,1/512],[15/512,12/512,16/512,13/512]])
        d=render_m1_grid(np.repeat(boxes,1050,axis=0),13,16,canvas=20)
        self.assertLess(abs(d.sum(dtype=np.float64)-2100),1e-4)
        self.assertEqual(float(d[13:,:].sum()+d[:,16:].sum()),0.)
        with self.assertRaises(ValueError):render_m1_grid([[0,0,1,1]],10,10)

    def test_sampler_repeats_only_dense_images_and_is_reproducible(self):
        names=['a','b','c'];counts={'a':299,'b':300,'c':1000}
        order=epoch_indices(names,counts,42,0)
        self.assertEqual(collections.Counter(order),{0:1,1:4,2:4})
        self.assertEqual(order,epoch_indices(names,counts,42,0))
        self.assertNotEqual(order,epoch_indices(names,counts,43,0))

    def test_rank_rule_adds_even_ranked_classes_only(self):
        names=[];counts={};labels={}
        for c,n in [('a',9),('b',8),('c',7),('d',3)]:
            for i in range(n):
                k=c+str(i);names.append(k);counts[k]=300;labels[k]=c
        old={'train_fit':names,'train_holdout':['h'],'train_fit_classes':['d','b','a','c']};counts['h']=1;labels['h']='h'
        fit,hold,added=expand_holdout(old,counts,labels)
        self.assertEqual(added,['b','d']);self.assertTrue(set(fit).isdisjoint(hold))
        self.assertEqual(len([n for n in hold if counts[n]>=300]),11)

    def test_safety_uses_epoch_ten_current_value_and_strict_comparison(self):
        self.assertFalse(safety_stop(9,100,10));self.assertTrue(safety_stop(10,10.01,10))
        self.assertFalse(safety_stop(10,10,10));self.assertFalse(safety_stop(11,100,10))
        with self.assertRaises(FloatingPointError):safety_stop(10,float('nan'),1)

    def test_diagnostic_thresholds_and_overcount_label_are_explicit(self):
        self.assertEqual(diagnostic_label(.699),'fit_problem_signal')
        self.assertEqual(diagnostic_label(.7),'mixed');self.assertEqual(diagnostic_label(.899),'mixed')
        self.assertEqual(diagnostic_label(.9),'generalization_problem_signal')
        self.assertEqual(diagnostic_label(2.),'generalization_problem_signal')

    def test_budget_import_counts_cancelled_work_once_and_rejects_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old.json';old.write_text(json.dumps({'state':'stopped','active':None,'entries':[
                {'elapsed_seconds':100},{'elapsed_seconds':20}]}))
            cfg={'predecessor_ledger':'old.json','predecessor_ledger_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'predecessor_spent_seconds':120}
            ledger=import_budget(root,root/'new.json',cfg);self.assertEqual(ledger.spent,120.)
            with self.assertRaises(FileExistsError):import_budget(root,root/'new.json',cfg)
            old.write_text(old.read_text()+' ')
            with self.assertRaises(ValueError):import_budget(root,root/'second.json',cfg)

    def test_registered_real_split_is_class_disjoint_and_val_unchanged(self):
        root=Path(__file__).resolve().parents[1];s=json.loads((root/'configs/m4_v2_splits.json').read_text());old=json.loads((root/'configs/round2_splits_v1.json').read_text())
        self.assertTrue(set(s['train_fit_classes']).isdisjoint(s['train_holdout_classes']))
        self.assertEqual(set(s['train_fit']+s['train_holdout']),set(old['train_fit']+old['train_holdout']))
        self.assertEqual(s['val_A'],old['val_A']);self.assertEqual(s['val_B'],old['val_B'])
        self.assertEqual(len(epoch_indices(s['train_fit'],s['counts'],42,0)),3359)


if __name__=='__main__':unittest.main()
