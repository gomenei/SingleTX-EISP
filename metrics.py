"""Per-object metrics accumulated without retaining the entire test set.

RRMSE and PSNR retain the original definitions. SSIM uses the original
independent per-image peak normalization, Gaussian window and zero padding.
IoU threshold defaults to 1.2, the original project's object threshold.
"""

import math

import torch
from pytorch_ssim import ssim


def ssim_2d(predicted, truth, grid_shape):
    peak_p = predicted.amax(dim=1, keepdim=True)
    peak_t = truth.amax(dim=1, keepdim=True)
    pred = predicted / torch.where(peak_p.abs() < 1e-12, torch.ones_like(peak_p), peak_p)
    true = truth / torch.where(peak_t.abs() < 1e-12, torch.ones_like(peak_t), peak_t)
    pred = pred.reshape(-1, 1, *reversed(grid_shape)).transpose(-1, -2)
    true = true.reshape(-1, 1, *reversed(grid_shape)).transpose(-1, -2)
    return ssim(pred, true, size_average=False)


class Metrics:
    def __init__(self, grid_shape, threshold=1.2):
        self.grid_shape = grid_shape
        self.threshold = threshold
        self.sums = {}
        self.count = 0

    def update(self, predicted, truth):
        pred, true = predicted.detach().double(), truth.detach().double()
        mse = (pred - true).square().mean(dim=1)
        rrmse = ((pred - true) / true.abs().clamp_min(1e-12)).square().mean(1).sqrt()
        peak = true.amax(1).abs().clamp_min(1e-12)
        psnr = 10 * torch.log10(peak.square() / mse)
        mask_p, mask_t = pred > self.threshold, true > self.threshold
        intersection = (mask_p & mask_t).sum(1)
        union = (mask_p | mask_t).sum(1)
        iou = torch.where(union == 0, torch.ones_like(mse), intersection / union.clamp_min(1))
        scores = dict(mse=mse, rrmse=rrmse, psnr=psnr, iou=iou)
        if len(self.grid_shape) == 2:
            scores["ssim"] = ssim_2d(predicted.detach(), truth.detach(), self.grid_shape).double()
        for key, values in scores.items():
            self.sums[key] = self.sums.get(key, 0.0) + values.sum().item()
        self.count += pred.shape[0]
        return {key: values.detach().cpu().numpy() for key, values in scores.items()}

    def result(self):
        if not self.count:
            raise ValueError("Cannot calculate metrics on an empty loader")
        return {key: value / self.count for key, value in self.sums.items()}


def finite_json(value):
    """Represent exact-match infinite PSNR as text, producing valid JSON."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
