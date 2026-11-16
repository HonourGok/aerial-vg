"""Failure analysis used in Section 7 of the report (aerial-vg).

Reads only results/<run>/per_sample.csv and results/<run>/train_log.txt, so it runs without data or GPU:

    python tools/analysis.py --results results --out results/analysis

Writes results/analysis/analysis.md and analysis.json with
  1. Pr@0.5 by object-size tercile (three equal groups by the rank of the ground-truth box area),
  2. word mentions among the smallest tercile,
  3. "stated side" correctness for expressions with only left / only right / only top-upper / only bottom-lower,
  4. disagreement counts vs. the reference run (McNemar b/c, losses with IoU < 0.3, wins from IoU < 0.3),
  5. per-epoch validation Pr@0.5 from the training logs.
"""
import argparse
import csv
import json
import os
import re
from math import comb

import numpy as np

RUNS = ['baseline', 'd4_remap', 'no_geo', 'd4_noremap']
COLS = ['iou', 'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']


def load(path):
    rows = sorted(csv.DictReader(open(os.path.join(path, 'per_sample.csv'))), key=lambda r: int(r['global_idx']))
    d = {c: np.array([float(r[c]) for r in rows]) for c in COLS}
    d['text'] = [r['text'].lower() for r in rows]
    d['idx'] = np.array([int(r['global_idx']) for r in rows])
    return d


def has(texts, pattern):
    rx = re.compile(pattern)
    return np.array([rx.search(t) is not None for t in texts])


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pct(x):
    return round(100 * float(np.mean(x)), 2) if len(x) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--ref', default='baseline')
    ap.add_argument('--prop', default='d4_remap')
    ap.add_argument('--out', default='results/analysis')
    a = ap.parse_args()
    runs = [r for r in RUNS if os.path.exists(os.path.join(a.results, r, 'per_sample.csv'))]
    D = {r: load(os.path.join(a.results, r)) for r in runs}
    ref = D[a.ref]
    for r in runs:
        assert (D[r]['idx'] == ref['idx']).all(), 'runs must cover the same test samples'
    out, md = {}, []

    # 1) size terciles by rank of GT box area (normalized coordinates)
    area = (ref['gt_x2'] - ref['gt_x1']) * (ref['gt_y2'] - ref['gt_y1'])
    rank = np.argsort(np.argsort(area, kind='stable'), kind='stable')
    n = len(area)
    grp = np.minimum(rank * 3 // n, 2)
    edges = [float(area[grp == g].max()) for g in (0, 1)]
    out['size_terciles'] = {'upper_edges_area_fraction': edges, 'n': [int((grp == g).sum()) for g in range(3)],
                            'pr05': {r: [pct(D[r]['iou'][grp == g] >= 0.5) for g in range(3)] for r in runs}}
    md += ['## Pr@0.5 by ground-truth box area (three equal groups by rank)', '',
           'Group edges (share of image area): small < %.2f%%, medium < %.2f%%' % (100 * edges[0], 100 * edges[1]), '',
           '| group | n | ' + ' | '.join(runs) + ' |', '|---|---|' + '---|' * len(runs)]
    for g, name in enumerate(['small', 'medium', 'large']):
        md.append('| %s | %d | ' % (name, (grp == g).sum()) +
                  ' | '.join('%.1f' % out['size_terciles']['pr05'][r][g] for r in runs) + ' |')

    # 2) word mentions in the smallest group
    small = [t for t, g in zip(ref['text'], grp) if g == 0]
    words = ['vehicle', 'windmill', 'bridge', 'ship', 'storage tank', 'airplane', 'chimney']
    out['small_group_mentions'] = {w: pct(has(small, r'\b%s' % w)) for w in words}
    md += ['', '## Share of small-group expressions that mention a class word (substring match)', '',
           ', '.join('%s %.1f%%' % kv for kv in out['small_group_mentions'].items())]

    # 3) stated-side correctness (centre of predicted / GT box in the stated half)
    t = ref['text']
    L, Rt = has(t, r'\bleft'), has(t, r'\bright')
    T, B = has(t, r'\b(top|upper)'), has(t, r'\b(bottom|lower)')
    subsets = {'left only': (L & ~Rt, 'x', 'lt'), 'right only': (Rt & ~L, 'x', 'gt'),
               'top/upper only': (T & ~B, 'y', 'lt'), 'bottom/lower only': (B & ~T, 'y', 'gt')}

    def side_ok(d, pre, axis, op):
        c = (d[pre + '_%s1' % axis] + d[pre + '_%s2' % axis]) / 2
        return c < 0.5 if op == 'lt' else c > 0.5
    out['stated_side'] = {}
    md += ['', '## Centre of the box lies in the half named by the expression (%)', '',
           '| subset | n | ground truth | ' + ' | '.join(runs) + ' |', '|---|---|---|' + '---|' * len(runs)]
    for name, (m, axis, op) in subsets.items():
        row = {'n': int(m.sum()), 'gt': pct(side_ok(ref, 'gt', axis, op)[m])}
        row.update({r: pct(side_ok(D[r], 'pred', axis, op)[m]) for r in runs})
        out['stated_side'][name] = row
        md.append('| %s | %d | %.1f | ' % (name, row['n'], row['gt']) + ' | '.join('%.1f' % row[r] for r in runs) + ' |')

    # 4) disagreements proposed vs reference
    p, q = D[a.prop]['iou'], ref['iou']
    b, c = int(((p >= .5) & (q < .5)).sum()), int(((p < .5) & (q >= .5)).sum())
    out['disagreement'] = {'prop_only_correct': b, 'ref_only_correct': c, 'mcnemar_p': mcnemar_exact(b, c),
                           'ref_correct_prop_below_0.3': int(((q >= .5) & (p < .3)).sum()),
                           'prop_correct_ref_below_0.3': int(((p >= .5) & (q < .3)).sum())}
    md += ['', '## %s vs %s at IoU 0.5' % (a.prop, a.ref), '', json.dumps(out['disagreement'])]

    # 5) validation curves
    out['val_pr05'] = {}
    md += ['', '## Validation Pr@0.5 per epoch (first 500 validation queries)', '']
    for r in runs:
        f = os.path.join(a.results, r, 'train_log.txt')
        if os.path.exists(f):
            logs = [json.loads(line) for line in open(f) if line.startswith('{')]
            v = [round(100 * x['validation_accu'], 1) for x in logs if 'validation_accu' in x]
            out['val_pr05'][r] = v
            md.append('- %s: %s' % (r, ' → '.join('%.1f' % x for x in v)))

    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'analysis.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'analysis.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
