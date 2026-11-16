"""Small CPU test of the D4 augmentation (no data, no GPU, a few seconds).

    python -m pytest -q tests        # or: python tests/test_geo_aug.py

Checks for every element of D4 that (1) the transformed box matches where the pixels actually move,
(2) a direction word in the text still points to the object's side after the transform, and
(3) the modes behave as described (hflip swaps only left/right, d4_noremap leaves the text unchanged).
"""
import importlib.util
import os
import random

import numpy as np
import torch
from PIL import Image

# load the module file directly so the test needs only torch, numpy and Pillow (not torchvision / transformers)
_spec = importlib.util.spec_from_file_location(
    'geo_aug', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'geo_aug.py'))
geo_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo_aug)
D4, RandomGeo, remap_text = geo_aug.D4, geo_aug.RandomGeo, geo_aug.remap_text
transform_box, transform_image = geo_aug.transform_box, geo_aug.transform_image

W, H = 64, 48
BOX = torch.tensor([5., 4., 20., 14.])     # an object in the upper-left corner


def image_with_box():
    a = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = BOX.int().tolist()
    a[y1:y2, x1:x2] = 255
    return Image.fromarray(a)


def test_box_follows_pixels():
    for op in D4:
        img = transform_image(image_with_box(), op)
        a = np.array(img)
        ys, xs = np.nonzero(a)
        pixel_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        box = transform_box(BOX, op, W, H).tolist()
        assert np.allclose(box, pixel_box), (op, box, pixel_box)


def side(box, w, h):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ('upper' if cy < h / 2 else 'lower') + ' ' + ('left' if cx < w / 2 else 'right')


def test_text_follows_geometry():
    text = 'the vehicle on the upper left'
    assert side(BOX.tolist(), W, H) == 'upper left'
    for op in D4:
        img = transform_image(image_with_box(), op)
        box = transform_box(BOX, op, W, H).tolist()
        assert remap_text(text, op) == 'the vehicle on the ' + side(box, img.width, img.height), op


def test_examples_from_report():
    assert remap_text('a vehicle on the upper left of the bridge', 'rot90') == 'a vehicle on the lower left of the bridge'
    assert remap_text('the ship at the top-right', 'transpose') == 'the ship at the bottom-left'
    assert remap_text('the leftmost storage tank', 'rot180') == 'the rightmost storage tank'
    assert remap_text('the airplane above the hangar', 'vflip') == 'the airplane below the hangar'
    assert remap_text('a large airport', 'rot270') == 'a large airport'


def test_modes():
    random.seed(0)
    d = {'img': image_with_box(), 'box': BOX.clone(), 'text': 'the ship on the upper left'}
    for _ in range(50):
        out = RandomGeo('d4_noremap')(dict(d))
        assert out['text'] == d['text']
        out = RandomGeo('hflip')(dict(d))
        assert out['text'] in ('the ship on the upper left', 'the ship on the upper right')
        out = RandomGeo('none')(dict(d))
        assert out['text'] == d['text'] and torch.equal(out['box'], BOX)


if __name__ == '__main__':
    for f in (test_box_follows_pixels, test_text_follows_geometry, test_examples_from_report, test_modes):
        f()
        print('ok', f.__name__)
