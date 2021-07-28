import sys
import os
import numpy as np
import pandas as pd

import cv2
from setuptools import glob

from imresize import imresize

sys.path.insert(0, "./PerceptualSimilarity")
from lpips import lpips
import tqdm

import torch

from skimage.metrics import peak_signal_noise_ratio as psnr

def fiFindByWildcard(wildcard):
    return glob.glob(os.path.expanduser(wildcard), recursive=True)


def dprint(d):
    out = []
    for k, v in d.items():
        out.append(f"{k}: {v:0.4f}")
    print(", ".join(out))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img[:, :, [2, 1, 0]]


loss_fn_alex_sp = lpips.LPIPS(spatial=True)

def lpips_analysis(gt, srs, scale):
    
    from collections import OrderedDict
    results = OrderedDict()

    gt = imread(gt)
    h, w, _ = gt.shape
    gt = gt[:(h//8)*8, :(w//8)*8]
    srs = [imread(sr) for sr in srs]

    lpipses_sp = []
    lpipses_gl = []
    lrpsnrs = []
    n_samples = len(srs)

    for sample_idx in range(n_samples):
        sr = srs[sample_idx]

        h1, w1, _ = gt.shape
        sr = sr[:h1, :w1]
        lpips_sp = loss_fn_alex_sp(2 * t(sr) - 1, 2 * t(gt) - 1)
        # print('----size', lpips_sp.shape)
        lpipses_sp.append(lpips_sp)
        lpipses_gl.append(lpips_sp.mean().item())

        imgA_lr = imresize(sr, 1 / scale)
        imgB_lr = imresize(gt, 1 / scale)
        lrpsnr = psnr(imgA_lr, imgB_lr)
        lrpsnrs.append(lrpsnr)

    lpips_gl = np.min(lpipses_gl)

    results['LPIPS_mean'] = np.mean(lpipses_gl)
    results['LRPSNR_worst'] = np.min(lrpsnrs)
    results['LRPSNR_mean'] = np.mean(lrpsnrs)

    lpipses_stacked = torch.stack([l[0, 0, :, :] for l in lpipses_sp], dim=2)

    lpips_best_sp, _ = torch.min(lpipses_stacked, dim=2)
    lpips_loc = lpips_best_sp.mean().item()

    score = (lpips_gl - lpips_loc) / lpips_gl * 100

    results['score'] = score

    dprint(results)

    return results





