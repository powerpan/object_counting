"""Preregistered nonnegative density heads and their scaled-density loss."""
import math
import torch
from torch import nn


class DensityHead(nn.Module):
    def __init__(self, channels, groups=32, initial_scaled_density=.01):
        super().__init__()
        blocks = []
        previous = 257
        for width in channels:
            blocks.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.GroupNorm(groups, width), nn.GELU()])
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Conv2d(previous, 1, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.constant_(self.final.bias, math.log(math.expm1(initial_scaled_density)))

    def forward(self, features, valid_mask):
        return torch.nn.functional.softplus(self.final(self.blocks(features))) * valid_mask


def density_loss(scaled_prediction, target, valid_mask, counts):
    # The network predicts Q=100*D. No second scaling of its output occurs.
    pixels = (((scaled_prediction - 100 * target) ** 2) * valid_mask).flatten(1).sum(1)
    pixels = pixels / valid_mask.flatten(1).sum(1).clamp_min(1)
    predicted_counts = scaled_prediction.flatten(1).sum(1) / 100
    count_error = (predicted_counts - counts).abs() / counts.clamp_min(1)
    loss = (pixels + count_error).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite density loss')
    return loss, {'pixel_mse': float(pixels.mean().detach()),
                  'relative_count_error': float(count_error.mean().detach()),
                  'count_mae': float((predicted_counts-counts).abs().mean().detach())}
