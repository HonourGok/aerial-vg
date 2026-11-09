"""Test-set evaluation for DIOR-RSVG with per-sample outputs (aerial-vg).

Reports Pr@{0.5..0.9}, meanIoU and cumIoU overall and for subsets
(expressions with / without direction words, horizontal-only, vertical words),
plus inference latency (batch 1) and peak GPU memory.

  python eval_dior.py --checkpoint outputs/baseline/best_checkpoint.pth \
      --data_root /path/DIOR_RSVG --split_root splits --out results/baseline
"""
import argparse
import csv
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.misc as utils
from datasets import build_dataset
from datasets.geo_aug import has_direction
from models import build_model
from utils.box_utils import xywh2xyxy

THRS = (0.5, 0.6, 0.7, 0.8, 0.9)
H_RE = re.compile(r'\b(left|right|leftmost|rightmost)\b')
V_RE = re.compile(r'\b(top|bottom|upper|lower|topmost|bottommost|uppermost|lowermost|above|below|under|beneath)\b')


def box_stats(pred, gt):
    """pred, gt: (N,4) xyxy. returns iou, inter, union."""
    lt = torch.max(pred[:, :2], gt[:, :2])
    rb = torch.min(pred[:, 2:], gt[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    a2 = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
    union = a1 + a2 - inter
    return inter / union.clamp(min=1e-9), inter, union


def summarize(iou, inter, union):
    if len(iou) == 0:
        return {'n': 0}
    out = {'n': int(len(iou))}
    for t in THRS:
        out['Pr@%.1f' % t] = float((iou >= t).float().mean() * 100)
    out['meanIoU'] = float(iou.mean() * 100)
    out['cumIoU'] = float(inter.sum() / union.sum() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split_root', default='splits')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--num_workers', default=2, type=int)
    ap.add_argument('--latency_samples', default=200, type=int)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    args = ckpt['args']
    args.data_root, args.split_root, args.device = a.data_root, a.split_root, a.device
    args.detr_model = None
    device = torch.device(a.device)

    model = build_model(args)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    ds = build_dataset(a.split, args)
    dl = DataLoader(ds, a.batch_size, shuffle=False, collate_fn=utils.collate_fn, num_workers=a.num_workers)

    preds, gts = [], []
    with torch.no_grad():
        for img_data, text_data, target in dl:
            out = model(img_data.to(device), text_data.to(device)).float().cpu()
            preds.append(out)
            gts.append(target)
    pred = torch.clamp(xywh2xyxy(torch.cat(preds)), 0, 1)
    gt = xywh2xyxy(torch.cat(gts))
    iou, inter, union = box_stats(pred, gt)

    texts = [t[2].lower() for t in ds.images]
    has_d = torch.tensor([has_direction(t) for t in texts])
    has_h = torch.tensor([H_RE.search(t) is not None for t in texts])
    has_v = torch.tensor([V_RE.search(t) is not None for t in texts])
    subsets = {'all': torch.ones_like(has_d), 'with_direction': has_d, 'no_direction': ~has_d,
               'horizontal_words': has_h, 'vertical_words': has_v}
    report = {k: summarize(iou[m], inter[m], union[m]) for k, m in subsets.items()}

    # latency, batch size 1, includes the full forward (visual + BERT + fusion)
    lat = []
    if a.latency_samples > 0:
        dl1 = DataLoader(ds, 1, shuffle=False, collate_fn=utils.collate_fn)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i, (img_data, text_data, _) in enumerate(dl1):
                if i >= a.latency_samples + 10:
                    break
                img_data, text_data = img_data.to(device), text_data.to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(img_data, text_data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if i >= 10:  # warm-up
                    lat.append((time.perf_counter() - t0) * 1000)
    report['latency_ms_median'] = float(np.median(lat)) if lat else None
    report['peak_infer_mem_gb'] = (torch.cuda.max_memory_allocated() / 2**30) if device.type == 'cuda' else None
    report['checkpoint'] = a.checkpoint
    report['epoch'] = int(ckpt.get('epoch', -1))
    report['geo_aug'] = getattr(args, 'geo_aug', 'hflip')
    report['train_fraction'] = getattr(args, 'train_fraction', 1.0)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(a.out, 'per_sample.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['global_idx', 'image', 'text', 'iou', 'has_direction', 'has_horizontal', 'has_vertical',
                    'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2'])
        for i, item in enumerate(ds.images):
            w.writerow([item[3], os.path.basename(item[0]), texts[i], '%.4f' % float(iou[i]),
                        int(has_d[i]), int(has_h[i]), int(has_v[i])] +
                       ['%.4f' % float(v) for v in pred[i]] + ['%.4f' % float(v) for v in gt[i]])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
