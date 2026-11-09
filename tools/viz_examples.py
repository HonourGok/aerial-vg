"""Draw predictions of two runs on the same test samples (aerial-vg).

Picks samples with direction words where the runs disagree (one hit, one miss)
and saves a grid: green = ground truth, red = run A, blue = run B.

  python tools/viz_examples.py --a results/baseline --b results/d4_remap \
      --data_root /path/DIOR_RSVG --out results/figures/disagreements.png
"""
import argparse
import csv
import os
import random

from PIL import Image, ImageDraw, ImageFont


def load(p):
    return {r['global_idx']: r for r in csv.DictReader(open(os.path.join(p, 'per_sample.csv')))}


def to_px(r, pre, w, h, s):
    # boxes are normalised to the padded square of side s that holds the resized image
    scale = s / max(w, h)
    dw, dh = (s - w * scale) / 2, (s - h * scale) / 2
    v = [float(r[pre + k]) * s for k in ('x1', 'y1', 'x2', 'y2')]
    return [(v[0] - dw) / scale, (v[1] - dh) / scale, (v[2] - dw) / scale, (v[3] - dh) / scale]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--imsize', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    keys = [k for k in A if A[k]['has_direction'] == '1' and
            (float(A[k]['iou']) >= 0.5) != (float(B[k]['iou']) >= 0.5)]
    random.Random(a.seed).shuffle(keys)
    keys = keys[:a.n]
    if not keys:
        print('no disagreements found')
        return
    tile = 400
    grid = Image.new('RGB', (tile * 4, (tile + 60) * ((len(keys) + 3) // 4)), 'white')
    font = ImageFont.load_default()
    for i, k in enumerate(keys):
        img = Image.open(os.path.join(a.data_root, 'JPEGImages', A[k]['image'])).convert('RGB')
        w, h = img.size
        d = ImageDraw.Draw(img)
        d.rectangle(to_px(A[k], 'gt_', w, h, a.imsize), outline=(0, 255, 0), width=4)
        d.rectangle(to_px(A[k], 'pred_', w, h, a.imsize), outline=(255, 0, 0), width=3)
        d.rectangle(to_px(B[k], 'pred_', w, h, a.imsize), outline=(0, 120, 255), width=3)
        img = img.resize((tile, tile))
        x, y = (i % 4) * tile, (i // 4) * (tile + 60)
        grid.paste(img, (x, y))
        ImageDraw.Draw(grid).text((x + 4, y + tile + 4), A[k]['text'][:60], fill='black', font=font)
        ImageDraw.Draw(grid).text((x + 4, y + tile + 24), 'IoU A=%s  B=%s' % (A[k]['iou'], B[k]['iou']),
                                  fill='black', font=font)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    grid.save(a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
