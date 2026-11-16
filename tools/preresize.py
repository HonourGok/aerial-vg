"""Resize DIOR-RSVG images once so the longer side is --size, and scale the XML boxes (aerial-vg).

Training and testing resize every image to --imsize anyway, so doing it once offline removes
repeated decoding/resizing of 800x800 JPEGs from the data loader (the CPU bottleneck on Kaggle).

  python tools/preresize.py --src /tmp/DIOR_RSVG --dst /tmp/DIOR_RSVG_640 --size 640
"""
import argparse
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

from PIL import Image


def do_image(args):
    src, dst, size = args
    im = Image.open(src).convert('RGB')
    w, h = im.size
    s = size / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    im.save(dst, quality=95)
    return os.path.basename(src), min(s, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.dst + '/JPEGImages', exist_ok=True)
    os.makedirs(a.dst + '/Annotations', exist_ok=True)
    imgs = sorted(f for f in os.listdir(a.src + '/JPEGImages') if f.lower().endswith('.jpg'))
    jobs = [(f'{a.src}/JPEGImages/{f}', f'{a.dst}/JPEGImages/{f}', a.size) for f in imgs]
    with ProcessPoolExecutor(a.workers) as ex:
        scale = dict(ex.map(do_image, jobs, chunksize=64))
    n = 0
    for f in sorted(os.listdir(a.src + '/Annotations')):
        if not f.endswith('.xml'):
            continue
        tree = ET.parse(f'{a.src}/Annotations/{f}')
        r = tree.getroot()
        s = scale[r.find('./filename').text]
        for member in r.findall('object'):
            bnd = member.find('bndbox')
            if bnd is None:
                bnd = member[2]
            for c in list(bnd)[:4]:
                c.text = str(int(round(float(c.text) * s)))
        size = r.find('size')
        if size is not None:
            for k in ('width', 'height'):
                e = size.find(k)
                if e is not None:
                    e.text = str(int(round(float(e.text) * s)))
        tree.write(f'{a.dst}/Annotations/{f}')
        n += 1
    print('resized %d images, rewrote %d xml files -> %s' % (len(imgs), n, a.dst))


if __name__ == '__main__':
    main()
