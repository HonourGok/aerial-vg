"""Mean and standard deviation over training seeds (aerial-vg).

Seed 13 is the main run (results/<name>); seeds 14 and 15 are results/<name>_s14, _s15.

    python tools/seed_summary.py --results results

Writes results/seeds/table.md and summary.json: per-seed Pr@0.5 / meanIoU / cumIoU for every condition, mean ± sample
sd over seeds, and the per-seed difference <condition> - baseline (paired by seed) with its mean ± sd.
"""
import argparse
import json
import os

import numpy as np

METRICS = ['Pr@0.5', 'meanIoU', 'cumIoU']
SUBSETS = ['all', 'with_direction', 'no_direction']


def find(results, name, seed):
    d = os.path.join(results, name if seed == 13 else '%s_s%d' % (name, seed), 'metrics.json')
    return json.load(open(d)) if os.path.exists(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[13, 14, 15])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(a.results, 'seeds')
    names = [n for n in ('baseline', 'd4_remap', 'd4_noremap', 'no_geo')
             if any(find(a.results, n, s) for s in a.seeds)]
    M = {n: {s: find(a.results, n, s) for s in a.seeds} for n in names}
    seeds = [s for s in a.seeds if all(M[n][s] for n in names)]
    if not seeds:
        print('no complete seed pairs found')
        return
    summary, md = {'seeds': seeds}, []
    for sub in SUBSETS:
        md += ['## %s' % sub, '', '| run | ' + ' | '.join('%s (seed %d)' % (m, s) for m in METRICS for s in seeds)
               + ' | ' + ' | '.join('%s mean ± sd' % m for m in METRICS) + ' |',
               '|---|' + '---|' * (len(METRICS) * (len(seeds) + 1))]
        for n in names:
            vals = {m: [M[n][s][sub][m] for s in seeds] for m in METRICS}
            summary.setdefault(sub, {})[n] = {m: {'per_seed': v, 'mean': float(np.mean(v)),
                                                  'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                              for m, v in vals.items()}
            md.append('| %s | ' % n + ' | '.join('%.2f' % x for m in METRICS for x in vals[m]) + ' | ' +
                      ' | '.join('%.2f ± %.2f' % (summary[sub][n][m]['mean'], summary[sub][n][m]['sd'])
                                 for m in METRICS) + ' |')
        for n in names[1:]:
            diff = {m: [M[n][s][sub][m] - M['baseline'][s][sub][m] for s in seeds] for m in METRICS}
            summary[sub]['diff_' + n] = {m: {'per_seed': v, 'mean': float(np.mean(v)),
                                             'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                         for m, v in diff.items()}
            d = summary[sub]['diff_' + n]
            md.append('| Δ %s − baseline | ' % n + ' | '.join('%+.2f' % x for m in METRICS for x in diff[m]) + ' | ' +
                      ' | '.join('%+.2f ± %.2f' % (d[m]['mean'], d[m]['sd']) for m in METRICS) + ' |')
        md.append('')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'table.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
