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

    def wordDropout(self, ids):
        if self.training == False or self.args.word_dropout == 0:
            return ids

        for i in range(ids.size(0)):
            for j in range(30, ids.size(1)):
                if ids[i, j] == 0:
                    break
                if numpy.random.rand() < self.args.word_dropout:
                    ids[i, j] = 100
        return ids

    def encode(self, ids, segments, mask):
        encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
        encoder = encoded_layers[-1][:, 0, :]
        return encoder.view(encoder.size(0), 1, -1)

    def forward(self, inputs):
        encoder1 = self.encode(inputs[0], inputs[1], inputs[2]) + self.encode(inputs[6], inputs[7], inputs[8])
        encoder2 = self.encode(inputs[3], inputs[4], inputs[5]) + self.encode(inputs[9], inputs[10], inputs[11])

        encoder = torch.cat([encoder1, encoder2], 1)
        scores = self.linear(encoder).view(-1, 2)
        return F.log_softmax(scores, dim=-1)

