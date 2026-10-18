import math
import unittest
import numpy as np
import torch
from planb.round2.data import point_target
from planb.round2.head import DensityHead, density_loss


class Round2DensityTests(unittest.TestCase):
    def test_target_mass_at_edges_and_overlapping_points(self):
        points=np.array([[0.,0.],[20.,18.],[5.,5.],[5.,5.]])
        d=point_target(points,18,20,2,canvas=24)
        self.assertAlmostEqual(float(d.sum()),4,places=12)
        self.assertEqual(float(d[18:,:].sum()+d[:,20:].sum()),0)
        self.assertTrue(np.isfinite(d).all())

    def test_empty_target_and_nonfinite_rejection(self):
        self.assertEqual(float(point_target([],8,9,1,canvas=10).sum()),0)
        with self.assertRaises(ValueError):point_target([[math.nan,0]],8,9,1,canvas=10)

    def test_scaled_density_loss_and_padding_are_consistent(self):
        target=torch.zeros(2,1,8,8);target[:,0,2,2]=1
        mask=torch.zeros_like(target);mask[:,:,:5,:6]=1
        q=(100*target).requires_grad_()
        loss,detail=density_loss(q,target,mask,torch.ones(2))
        self.assertEqual(float(loss),0.)
        self.assertEqual(detail['count_mae'],0.)
        loss.backward();self.assertTrue(torch.isfinite(q.grad).all())

    def test_declared_head_sizes_and_nonnegative_initial_mass(self):
        torch.set_num_threads(2)
        for channels,expected in [([256,256,256,128,64],2142401),([512,512,512,256,128],7381377)]:
            h=DensityHead(channels)
            self.assertEqual(sum(p.numel() for p in h.parameters()),expected)
            x=torch.zeros(1,257,8,8);mask=torch.zeros(1,1,8,8);mask[:,:,:4,:5]=1
            q=h(x,mask)
            self.assertTrue((q>=0).all())
            self.assertAlmostEqual(float(q.sum().detach()),.2,places=6)
            self.assertEqual(float(q[:,:,4:,:].sum()),0.)


if __name__=='__main__':unittest.main()
