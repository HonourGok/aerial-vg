"""Paired comparison of runs on the same test samples (aerial-vg).

  python compare.py --runs baseline=results/baseline d4_remap=results/d4_remap d4_noremap=results/d4_noremap \
      --ref baseline --out results/comparison

For each run vs. the reference: Pr@0.5 difference with a paired bootstrap 95% CI
and an exact McNemar test on the hit/miss tables, overall and per subset.
Also writes a markdown table (results/comparison/table.md).
"""
import argparse
import csv
import json
import os
from math import comb

import numpy as np

SUBSETS = {'all': None, 'with_direction': 'has_direction', 'no_direction': '!has_direction',
           'horizontal_words': 'has_horizontal', 'vertical_words': 'has_vertical'}


def load(path):
    rows = list(csv.DictReader(open(os.path.join(path, 'per_sample.csv'))))
    rows.sort(key=lambda r: int(r['global_idx']))
    return rows


def mask(rows, spec):
    if spec is None:
        return np.ones(len(rows), bool)
    neg = spec.startswith('!')
    col = spec.lstrip('!')
    m = np.array([r[col] == '1' for r in rows])
    return ~m if neg else m


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='name=results_dir')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    runs = dict(r.split('=', 1) for r in a.runs)
    data = {k: load(v) for k, v in runs.items()}
    ids = [r['global_idx'] for r in data[a.ref]]
    for k, rows in data.items():
        assert [r['global_idx'] for r in rows] == ids, 'runs were evaluated on different samples: %s' % k
    hits = {k: np.array([float(r['iou']) >= 0.5 for r in rows]) for k, rows in data.items()}
    ious = {k: np.array([float(r['iou']) for r in rows]) for k, rows in data.items()}
    rng = np.random.default_rng(a.seed)
    res = {}
    for sname, spec in SUBSETS.items():
        m = mask(data[a.ref], spec)
        idx = np.nonzero(m)[0]
        res[sname] = {'n': int(len(idx))}
        for k in data:
            res[sname][k] = {'Pr@0.5': float(hits[k][idx].mean() * 100) if len(idx) else None,
                             'meanIoU': float(ious[k][idx].mean() * 100) if len(idx) else None}
            if k == a.ref or len(idx) == 0:
                continue
            d = hits[k][idx].astype(float) - hits[a.ref][idx].astype(float)
            boots = [d[rng.integers(0, len(idx), len(idx))].mean() for _ in range(a.n_boot)]
            b = int(np.sum(hits[k][idx] & ~hits[a.ref][idx]))
            c = int(np.sum(~hits[k][idx] & hits[a.ref][idx]))
            res[sname][k]['delta_Pr@0.5_vs_ref'] = float(d.mean() * 100)
            res[sname][k]['ci95'] = [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)]
            res[sname][k]['mcnemar_p'] = mcnemar_exact(b, c)
            res[sname][k]['wins_losses'] = [b, c]
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'comparison.json'), 'w'), indent=2)

    lines = ['| Subset | n | ' + ' | '.join('%s Pr@0.5' % k for k in data) + ' | ' +
             ' | '.join('Δ %s vs %s [95%% CI], p' % (k, a.ref) for k in data if k != a.ref) + ' |']
    lines.append('|' + '---|' * (2 + len(data) + len(data) - 1))
    for sname in SUBSETS:
        r = res[sname]
        if r['n'] == 0:
            continue
        cells = ['%.2f' % r[k]['Pr@0.5'] for k in data]
        deltas = ['%+.2f [%+.2f, %+.2f], p=%.3g' % (r[k]['delta_Pr@0.5_vs_ref'], *r[k]['ci95'], r[k]['mcnemar_p'])
                  for k in data if k != a.ref]
        lines.append('| %s | %d | %s | %s |' % (sname, r['n'], ' | '.join(cells), ' | '.join(deltas)))
    open(os.path.join(a.out, 'table.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
