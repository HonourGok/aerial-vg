"""Aerial-aware geometric augmentation with direction-word remapping (aerial-vg).

Overhead imagery has no canonical "up", so all 8 symmetries of the square
(the dihedral group D4: identity, flips, 90-degree rotations, transposes) are
plausible views of the same scene. A referring expression such as
"the ship at the upper left" is only still correct after a geometric transform
if its direction words are rewritten consistently.

Modes (see datasets/__init__.py, flag --geo_aug):
  hflip      : TransVG default. Horizontal flip with p=0.5, left<->right swap.
  d4_remap   : PROPOSED. Uniformly random D4 element; direction words remapped.
  d4_noremap : CONTROL. Same random D4 element, text left unchanged
               (labels become inconsistent whenever a direction word is present).
  none       : no geometric augmentation.
"""
import random
import re

import torch
from PIL import Image

# ---------------------------------------------------------------- geometry
# direction unit vectors in image coordinates (x to the right, y downwards)
DIRS = {'L': (-1, 0), 'R': (1, 0), 'T': (0, -1), 'B': (0, 1)}
VEC2DIR = {v: k for k, v in DIRS.items()}

# name -> (PIL op or None, point map (x, y, w, h) -> (x', y'), vector map (dx, dy) -> (dx', dy'))
# PIL ROTATE_90 rotates counter-clockwise; output size is (h, w).
D4 = {
    'id':        (None,                       lambda x, y, w, h: (x, y),         lambda dx, dy: (dx, dy)),
    'hflip':     (Image.FLIP_LEFT_RIGHT,      lambda x, y, w, h: (w - x, y),     lambda dx, dy: (-dx, dy)),
    'vflip':     (Image.FLIP_TOP_BOTTOM,      lambda x, y, w, h: (x, h - y),     lambda dx, dy: (dx, -dy)),
    'rot90':     (Image.ROTATE_90,            lambda x, y, w, h: (y, w - x),     lambda dx, dy: (dy, -dx)),
    'rot180':    (Image.ROTATE_180,           lambda x, y, w, h: (w - x, h - y), lambda dx, dy: (-dx, -dy)),
    'rot270':    (Image.ROTATE_270,           lambda x, y, w, h: (h - y, x),     lambda dx, dy: (-dy, dx)),
    'transpose': (Image.TRANSPOSE,            lambda x, y, w, h: (y, x),         lambda dx, dy: (dy, dx)),
    'transverse': (Image.TRANSVERSE,          lambda x, y, w, h: (h - y, w - x), lambda dx, dy: (-dy, -dx)),
}


def dir_map(op_name):
    vmap = D4[op_name][2]
    return {d: VEC2DIR[vmap(*v)] for d, v in DIRS.items()}


def transform_box(box, op_name, w, h):
    """box: tensor [x1, y1, x2, y2] in pixels of the (w, h) image."""
    pmap = D4[op_name][1]
    x1, y1, x2, y2 = [float(v) for v in box]
    pts = [pmap(x, y, w, h) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return torch.tensor([min(xs), min(ys), max(xs), max(ys)], dtype=box.dtype)


def transform_image(img, op_name):
    op = D4[op_name][0]
    return img if op is None else img.transpose(op)


# ---------------------------------------------------------------- language
# word -> (direction, word family). Families keep the part of speech stable.
WORDS = {
    'left': ('L', 'side'), 'right': ('R', 'side'), 'top': ('T', 'side'), 'bottom': ('B', 'side'),
    'upper': ('T', 'adj'), 'lower': ('B', 'adj'),
    'leftmost': ('L', 'most'), 'rightmost': ('R', 'most'), 'topmost': ('T', 'most'),
    'bottommost': ('B', 'most'), 'uppermost': ('T', 'most'), 'lowermost': ('B', 'most'),
    'above': ('T', 'prep'), 'below': ('B', 'prep'), 'under': ('B', 'prep'), 'beneath': ('B', 'prep'),
}
FAMILY_OUT = {
    'side': {'L': 'left', 'R': 'right', 'T': 'top', 'B': 'bottom'},
    'adj':  {'L': 'left', 'R': 'right', 'T': 'upper', 'B': 'lower'},
    'most': {'L': 'leftmost', 'R': 'rightmost', 'T': 'topmost', 'B': 'bottommost'},
    'prep': {'L': 'left of', 'R': 'right of', 'T': 'above', 'B': 'below'},
}
VERTICAL = {'T', 'B'}
_word_re = re.compile(r'\b(' + '|'.join(sorted(WORDS, key=len, reverse=True)) + r')\b')
# compound corners like "upper left", "top-right", "lower right"
_corner_re = re.compile(r'\b(upper|lower|top|bottom)([\s-]+)(left|right)\b')

HORIZONTAL_ONLY = re.compile(r'\b(left|right|leftmost|rightmost)\b')


def has_direction(text):
    return _word_re.search(text) is not None


def remap_text(text, op_name):
    """Rewrite direction words so the expression stays true after transform op_name."""
    if op_name == 'id':
        return text
    m = dir_map(op_name)

    def corner(match):
        v_word, sep, h_word = match.group(1), match.group(2), match.group(3)
        d1, d2 = m[WORDS[v_word][0]], m[WORDS[h_word][0]]
        # keep the conventional order: vertical word first
        vert, horiz = (d1, d2) if d1 in VERTICAL else (d2, d1)
        fam = WORDS[v_word][1]
        return FAMILY_OUT[fam][vert] + sep + FAMILY_OUT['side'][horiz]

    placeholder = []

    def stash(s):
        placeholder.append(s)
        return '\u0000%d\u0000' % (len(placeholder) - 1)

    text = _corner_re.sub(lambda mt: stash(corner(mt)), text)

    def single(match):
        d, fam = WORDS[match.group(1)]
        return FAMILY_OUT[fam][m[d]]

    text = _word_re.sub(single, text)
    for i, s in enumerate(placeholder):
        text = text.replace('\u0000%d\u0000' % i, s)
    # "left of of" can appear when 'above' maps to 'left of' before an existing 'of'
    text = re.sub(r'\b(left|right) of of\b', r'\1 of', text)
    return text


# ---------------------------------------------------------------- transform
class RandomGeo(object):
    """Drop-in replacement for TransVG's RandomHorizontalFlip (works on the input_dict)."""

    def __init__(self, mode='hflip'):
        assert mode in ('hflip', 'd4_remap', 'd4_noremap', 'none')
        self.mode = mode

    def sample_op(self):
        if self.mode == 'hflip':
            return 'hflip' if random.random() < 0.5 else 'id'
        if self.mode == 'none':
            return 'id'
        return random.choice(list(D4.keys()))

    def __call__(self, input_dict):
        op = self.sample_op()
        if op == 'id':
            return input_dict
        img, box, text = input_dict['img'], input_dict['box'], input_dict['text']
        w, h = img.width, img.height
        input_dict['img'] = transform_image(img, op)
        input_dict['box'] = transform_box(box, op, w, h)
        if self.mode != 'd4_noremap':
            if self.mode == 'hflip':
                # exactly TransVG's original rule (substring swap of left/right only)
                text = text.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
            else:
                text = remap_text(text, op)
        input_dict['text'] = text
        return input_dict
