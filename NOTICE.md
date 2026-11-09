# Third-party code and data

- **TransVG** (Deng et al., ICCV 2021), https://github.com/djiajunustc/TransVG, commit
  `c8624277018fc9786a39181e229f155f933b2b59`. `models/`, `utils/`, `engine.py`, `train.py` and
  `datasets/transforms.py`, `datasets/data_loader.py` are adapted from it. The upstream repository does not ship a
  license file; the code is used here for non-commercial academic research with attribution.
  TransVG itself builds on DETR (Apache-2.0, Facebook) and pytorch-pretrained-BERT.
- **RSVG / DIOR-RSVG** (Zhan et al., IEEE TGRS 2023), https://github.com/ZhanYang-nwpu/RSVG-pytorch, commit
  `f9e1f901d34d0aeee250caea1d42e0eaac505fb0`, CC BY-NC 4.0. `splits/*.txt` are copied from it, and the XML indexing in
  `datasets/dior_rsvg.py` follows its `data_loader.py`. The dataset is research-only and is **not** redistributed here.
- **DETR-R50 COCO weights**: https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth (downloaded at run time).
- **BERT-base-uncased**: downloaded from the Hugging Face Hub at run time.
