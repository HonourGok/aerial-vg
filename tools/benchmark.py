"""Measure data-loading and training/eval throughput before launching long runs (aerial-vg).

  python tools/benchmark.py --data_root $DATA_ROOT --detr_model $DETR --iters 40 --num_workers 4

Prints samples/s for (1) the training data loader alone, (2) full training steps,
(3) eval forward passes, and an estimate of the time per epoch for --train_fraction.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import utils.misc as utils  # noqa: E402
import utils.loss_utils as loss_utils  # noqa: E402
from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from train import get_args_parser  # noqa: E402


def cycle(loader):
    while True:
        for b in loader:
            yield b


def rate(loader, n, fn):
    it = cycle(loader)
    fn(next(it))  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0, k = time.time(), 0
    for _ in range(n):
        b = next(it)
        fn(b)
        k += b[0].tensors.shape[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return k / (time.time() - t0)


def main():
    p = get_args_parser()
    p.add_argument('--iters', type=int, default=40)
    args = p.parse_args()
    args.dataset = 'dior_rsvg'
    args.distributed = False
    dev = torch.device(args.device)
    print('cpus:', os.cpu_count(), '| gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    ds_tr = build_dataset('train', args)
    ds_te = build_dataset('test', args)
    dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True, drop_last=True, collate_fn=utils.collate_fn,
                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    dl_te = DataLoader(ds_te, args.batch_size, shuffle=False, collate_fn=utils.collate_fn,
                       num_workers=args.num_workers)

    data_rate = rate(dl_tr, args.iters, lambda b: None)
    print('1) train data loader only : %.1f samples/s' % data_rate)

    model = build_model(args).to(dev)
    if args.detr_model:
        model.visumodel.load_state_dict(torch.load(args.detr_model, map_location='cpu', weights_only=False)['model'],
                                        strict=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    model.train()

    def step(b):
        img, txt, tgt = [x.to(dev) for x in b]
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            out = model(img, txt)
        loss = sum(loss_utils.trans_vg_loss(out.float(), tgt).values())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_rate = rate(dl_tr, args.iters, step)
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print('2) full training step      : %.1f samples/s  (peak mem %.1f GB)' % (train_rate, mem))

    model.eval()

    @torch.no_grad()
    def fwd(b):
        img, txt, _ = [x.to(dev) for x in b]
        model(img, txt)
    eval_rate = rate(dl_te, args.iters, fwd)
    print('3) eval forward            : %.1f samples/s' % eval_rate)

    n_tr = len(ds_tr)
    print('--> one epoch on %d train samples: %.1f min (+ val %d samples: %.1f min); full test: %.1f min'
          % (n_tr, n_tr / train_rate / 60, args.val_max, args.val_max / eval_rate / 60, 7500 / eval_rate / 60))
    print('BOTTLENECK:', 'DATA LOADING (CPU)' if data_rate < 1.3 * train_rate else 'GPU compute')


if __name__ == '__main__':
    main()
