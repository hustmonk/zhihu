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
        proj = 256
        self.linear = nn.Linear(args.embedding_dim, proj)

        self.answer_self_attn = layers.LinearSeqAttn(proj)
        self.passage_self_attn = layers.LinearSeqAttn(proj)

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

        ids, segments, mask, core, passage_mask = info

        encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
        last_encoded_layer = encoded_layers[-1]
        last_encoded_layer = F.dropout(last_encoded_layer, p=0.2, training=self.training)

        last_encoded_layers = self.linear(last_encoded_layer)
        answer_encoder = self.answer_self_attn(last_encoded_layers, core).unsqueeze(1)
        passage_encoder = self.passage_self_attn(last_encoded_layers, passage_mask)

        return answer_encoder, passage_encoder

    def forward(self, inputs):
        encoder1, passage_encoder1 = self.encode(inputs[0])
        encoder2, passage_encoder2 = self.encode(inputs[1])

        passage_encoder = (passage_encoder1 + passage_encoder2)/2
        answer1 = encoder1.bmm(passage_encoder.unsqueeze(2)).squeeze(2)
        answer2 = encoder2.bmm(passage_encoder.unsqueeze(2)).squeeze(2)

        scores = torch.cat([answer1, answer2], 1)

        return F.log_softmax(scores, dim=-1)

