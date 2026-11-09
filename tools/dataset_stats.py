"""Direction-word statistics of DIOR-RSVG expressions (aerial-vg).

  python tools/dataset_stats.py --data_root /path/DIOR_RSVG --split_root splits --out results/dataset_stats.json
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dior_rsvg import load_dior_rsvg  # noqa: E402
from datasets.geo_aug import WORDS, has_direction, remap_text  # noqa: E402

TOKEN = re.compile(r"[a-z]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split_root', default='splits')
    ap.add_argument('--out', default='results/dataset_stats.json')
    a = ap.parse_args()
    out = {}
    for split in ('train', 'val', 'test'):
        items = load_dior_rsvg(a.data_root, os.path.join(a.split_root, split + '.txt'))
        texts = [t[2].lower() for t in items]
        words = collections.Counter(w for t in texts for w in TOKEN.findall(t))
        out[split] = {
            'n_expressions': len(texts),
            'n_images': len(set(t[0] for t in items)),
            'pct_with_direction_word': 100.0 * sum(has_direction(t) for t in texts) / max(1, len(texts)),
            'direction_word_counts': {w: words[w] for w in WORDS if words[w]},
            'mean_words': sum(len(TOKEN.findall(t)) for t in texts) / max(1, len(texts)),
            'top_words': words.most_common(60),
        }
    # examples of the remapping, for the report appendix
    train = load_dior_rsvg(a.data_root, os.path.join(a.split_root, 'train.txt'))
    ex = [t[2].lower() for t in train if has_direction(t[2].lower())][:10]
    out['remap_examples'] = [{op: remap_text(t, op) for op in ('id', 'hflip', 'vflip', 'rot90', 'transpose')} for t in ex]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    for s in ('train', 'val', 'test'):
        print(s, {k: v for k, v in out[s].items() if k != 'top_words'})


if __name__ == '__main__':
    main()
