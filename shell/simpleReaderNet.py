import torch
import torch.nn as nn
import logging
from shell import layers
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import numpy

logger = logging.getLogger(__name__)

class ReaderNet(nn.Module):
    def __init__(self, args):
        super(ReaderNet, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.merge = 4
        self.linear = nn.Linear(int(args.embedding_dim), 1)
        self.answer_self_attn = layers.LinearSeqAttn(args.embedding_dim)

        self.maskid = 100

    def wordDropout(self, ids):
        if self.training == False or self.args.word_dropout == 0:
            return ids

        for i in range(ids.size(0)):
            for j in range(30, ids.size(1)):
                if ids[i, j] == 0:
                    break
                if numpy.random.rand() < self.args.word_dropout:
                    ids[i, j] = self.maskid
        return ids

    def encode(self, info):
        ids, segments, mask, core = info

        encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
        self_encoder = self.answer_self_attn(encoded_layers[-1], core)
        #encoder = encoded_layers[-1][:, 0, :] + self_encoder
        encoder = self_encoder
        return encoder.view(encoder.size(0), 1, -1)

    def forward(self, inputs):
        encoder1 = self.encode(inputs[0]) + self.encode(inputs[2])
        encoder2 = self.encode(inputs[1]) + self.encode(inputs[3])

        encoder = torch.cat([encoder1, encoder2], 1)
        scores = self.linear(encoder).view(-1, 2)
        return F.log_softmax(scores, dim=-1)

