"""Mean and standard deviation over training seeds (aerial-vg).

Seed 13 is the main run (results/baseline, results/d4_remap); seeds 14 and 15 are results/<name>_s14, _s15.

    python tools/seed_summary.py --results results

Writes results/seeds/table.md and summary.json: per-seed Pr@0.5 / meanIoU / cumIoU, mean ± sample sd over seeds,
and the per-seed difference d4_remap - baseline (paired by seed) with its mean ± sd.
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
    M = {n: {s: find(a.results, n, s) for s in a.seeds} for n in ('baseline', 'd4_remap')}
    seeds = [s for s in a.seeds if M['baseline'][s] and M['d4_remap'][s]]
    if not seeds:
        print('no complete seed pairs found')
        return
    summary, md = {'seeds': seeds}, []
    for sub in SUBSETS:
        md += ['## %s' % sub, '', '| run | ' + ' | '.join('%s (seed %d)' % (m, s) for m in METRICS for s in seeds)
               + ' | ' + ' | '.join('%s mean ± sd' % m for m in METRICS) + ' |',
               '|---|' + '---|' * (len(METRICS) * (len(seeds) + 1))]
        for n in ('baseline', 'd4_remap'):
            vals = {m: [M[n][s][sub][m] for s in seeds] for m in METRICS}
            summary.setdefault(sub, {})[n] = {m: {'per_seed': v, 'mean': float(np.mean(v)),
                                                  'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                              for m, v in vals.items()}
            md.append('| %s | ' % n + ' | '.join('%.2f' % x for m in METRICS for x in vals[m]) + ' | ' +
                      ' | '.join('%.2f ± %.2f' % (summary[sub][n][m]['mean'], summary[sub][n][m]['sd'])
                                 for m in METRICS) + ' |')
        diff = {m: [M['d4_remap'][s][sub][m] - M['baseline'][s][sub][m] for s in seeds] for m in METRICS}
        summary[sub]['diff'] = {m: {'per_seed': v, 'mean': float(np.mean(v)),
                                    'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for m, v in diff.items()}
        md.append('| Δ d4_remap − baseline | ' + ' | '.join('%+.2f' % x for m in METRICS for x in diff[m]) + ' | ' +
                  ' | '.join('%+.2f ± %.2f' % (summary[sub]['diff'][m]['mean'], summary[sub]['diff'][m]['sd'])
                             for m in METRICS) + ' |')
        md.append('')
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'table.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
