# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

from transformers import BertModel  # (aerial-vg) ported from pytorch_pretrained_bert


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        self.enc_num = enc_num

        self.bert = BertModel.from_pretrained(name)
        self.num_channels = self.bert.config.hidden_size

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):

        if self.enc_num > 0:
            out = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask,
                            output_hidden_states=True)
            # use the output of the X-th transformer encoder layer.
            # HF hidden_states[0] is the embedding output, so layer k is hidden_states[k]
            # (pytorch_pretrained_bert's all_encoder_layers[k-1]).
            xs = out.hidden_states[self.enc_num]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
