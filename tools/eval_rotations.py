"""Orientation-robustness test: evaluate trained models on transformed copies of the test set (aerial-vg).

Every test image is flipped/rotated by one fixed D4 transform; by default the expression is remapped
so it stays true (see datasets/geo_aug.py). No training is involved.

  python tools/eval_rotations.py --data_root $DATA_ROOT \
      --runs baseline d4_remap d4_noremap no_geo --ops id rot90 rot180 rot270 vflip

Writes results/<run>/rot_<op>/{metrics.json,per_sample.csv}, then
results/robustness/{robustness.csv,table.md} and a paired comparison per transform
(results/robustness/compare_<op>/) against --ref.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSETS = ('all', 'with_direction', 'no_direction', 'horizontal_words', 'vertical_words')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split_root', default='splits')
    ap.add_argument('--runs', nargs='+', default=['baseline', 'd4_remap', 'd4_noremap', 'no_geo'])
    ap.add_argument('--ops', nargs='+', default=['id', 'rot90', 'rot180', 'rot270', 'vflip'])
    ap.add_argument('--ref', default='baseline')
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--results', default='results')
    ap.add_argument('--no_remap', action='store_true', help='keep original text (tests image-only robustness)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--num_workers', default=2, type=int)
    ap.add_argument('--skip_existing', action='store_true')
    a = ap.parse_args()

    tag = 'rot' if not a.no_remap else 'rotnoremap'
    for run in a.runs:
        ckpt = os.path.join(a.outputs, run, 'best_checkpoint.pth')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(a.outputs, run, 'checkpoint.pth')
        if not os.path.exists(ckpt):
            print('skip %s: no checkpoint' % run)
            continue
        for op in a.ops:
            out = os.path.join(a.results, run, '%s_%s' % (tag, op))
            if a.skip_existing and os.path.exists(os.path.join(out, 'metrics.json')):
                continue
            cmd = [sys.executable, os.path.join(ROOT, 'eval_dior.py'), '--checkpoint', ckpt,
                   '--data_root', a.data_root, '--split_root', a.split_root, '--out', out,
                   '--test_op', op, '--latency_samples', '0', '--num_workers', str(a.num_workers)]
            if a.no_remap:
                cmd.append('--no_test_remap')
            if a.device:
                cmd += ['--device', a.device]
            print('>>', run, op, flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    # summary table
    rows = []
    for run in a.runs:
        for op in a.ops:
            p = os.path.join(a.results, run, '%s_%s' % (tag, op), 'metrics.json')
            if not os.path.exists(p):
                continue
            m = json.load(open(p))
            for s in SUBSETS:
                if m.get(s, {}).get('n'):
                    rows.append({'run': run, 'op': op, 'subset': s, 'n': m[s]['n'],
                                 'pr50': round(m[s]['Pr@0.5'], 2), 'miou': round(m[s]['meanIoU'], 2)})
    outdir = os.path.join(a.results, 'robustness' if not a.no_remap else 'robustness_noremap')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'robustness.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'op', 'subset', 'n', 'pr50', 'miou'])
        w.writeheader()
        w.writerows(rows)

    def get(run, op, s='all'):
        for r in rows:
            if r['run'] == run and r['op'] == op and r['subset'] == s:
                return r['pr50']
        return None

    lines = []
    for s in ('all', 'with_direction'):
        lines.append('\n**Pr@0.5 (%%), subset: %s**\n' % s)
        lines.append('| run | ' + ' | '.join(a.ops) + ' | mean over non-id | drop vs id |')
        lines.append('|---' * (len(a.ops) + 3) + '|')
        for run in a.runs:
            vals = [get(run, op, s) for op in a.ops]
            if all(v is None for v in vals):
                continue
            non_id = [v for op, v in zip(a.ops, vals) if op != 'id' and v is not None]
            mean_non = sum(non_id) / len(non_id) if non_id else None
            base = get(run, 'id', s)
            drop = (mean_non - base) if (mean_non is not None and base is not None) else None
            cell = lambda v: '-' if v is None else '%.2f' % v
            lines.append('| %s | %s | %s | %s |' % (run, ' | '.join(cell(v) for v in vals), cell(mean_non),
                                                    '-' if drop is None else '%+.2f' % drop))
    open(os.path.join(outdir, 'table.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # paired tests per transform against the reference run
    for op in a.ops:
        present = [r for r in a.runs if os.path.exists(os.path.join(a.results, r, '%s_%s' % (tag, op), 'per_sample.csv'))]
        if a.ref in present and len(present) > 1:
            subprocess.run([sys.executable, os.path.join(ROOT, 'compare.py'), '--runs'] +
                           ['%s=%s' % (r, os.path.join(a.results, r, '%s_%s' % (tag, op))) for r in present] +
                           ['--ref', a.ref, '--out', os.path.join(outdir, 'compare_%s' % op)],
                           check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
