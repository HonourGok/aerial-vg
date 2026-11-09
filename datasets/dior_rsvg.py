"""DIOR-RSVG dataset for the TransVG pipeline (aerial-vg).

Indexing follows the official RSVG loader (github.com/ZhanYang-nwpu/RSVG-pytorch,
data_loader.py): every <object> of every XML file in sorted order gets a running
index, and splits/{train,val,test}.txt list the indices belonging to each split.
"""
import os
import os.path as osp
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from transformers import BertTokenizer

from .data_loader import read_examples, convert_examples_to_features


def _obj_fields(member):
    """Return (box[x1,y1,x2,y2], text). Uses tag names when present, else the
    positional layout used by the official loader (member[2]=bndbox, member[3]=text)."""
    bnd = member.find('bndbox')
    if bnd is None:
        bnd = member[2]
        box = [int(float(bnd[i].text)) for i in range(4)]
    else:
        box = [int(float(bnd.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
    desc = member.find('description')
    text = desc.text if desc is not None else member[3].text
    return np.array(box, dtype=np.float32), (text or '').strip()


def load_dior_rsvg(root, split_file):
    ann_dir = osp.join(root, 'Annotations')
    img_dir = osp.join(root, 'JPEGImages')
    wanted = set(int(l.strip()) for l in open(split_file) if l.strip())
    xmls = sorted(f for f in os.listdir(ann_dir) if f.endswith('.xml'))
    items, count = [], 0
    for f in xmls:
        r = ET.parse(osp.join(ann_dir, f)).getroot()
        fname = r.find('./filename').text
        for member in r.findall('object'):
            if count in wanted:
                box, text = _obj_fields(member)
                items.append((osp.join(img_dir, fname), box, text, count))
            count += 1
    return items


class DIORRSVGDataset(data.Dataset):
    def __init__(self, data_root, split_dir, split='train', transform=None, max_query_len=40,
                 bert_model='bert-base-uncased', fraction=1.0, seed=13):
        self.transform = transform
        self.query_len = max_query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.images = load_dior_rsvg(data_root, osp.join(split_dir, split + '.txt'))
        if fraction < 1.0:
            rng = random.Random(seed)  # fixed subset, identical for every run with the same seed
            k = max(1, int(round(len(self.images) * fraction)))
            self.images = sorted(rng.sample(self.images, k), key=lambda t: t[3])
        if len(self.images) == 0:
            raise RuntimeError('No samples for split %s - check --data_root and splits/' % split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, box, phrase, _ = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        input_dict = {'img': img, 'box': torch.tensor(box).float(), 'text': phrase.lower()}
        input_dict = self.transform(input_dict)
        examples = read_examples(input_dict['text'], idx)
        feats = convert_examples_to_features(examples=examples, seq_length=self.query_len,
                                             tokenizer=self.tokenizer)
        return (input_dict['img'], np.array(input_dict['mask']),
                np.array(feats[0].input_ids, dtype=int), np.array(feats[0].input_mask, dtype=int),
                np.array(input_dict['box'], dtype=np.float32))
