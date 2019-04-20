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
        self.linear = nn.Linear(args.embedding_dim * 2, 1)

    def wordDropout(self, ids):
        if self.training == False and self.args.word_dropout > 0:
            return ids

        for i in range(ids.size(0)):
            for j in range(ids.size(1)):
                if ids[i, j] == 0:
                    break
                if numpy.random.rand() < self.args.word_dropout:
                    ids[i, j] = 100
        return ids

    def forward(self, inputs):
        scores = []
        for (i) in range(0, len(inputs), 3):
            ids = self.wordDropout(inputs[i])
            segments = inputs[i + 1]
            mask = inputs[i + 2]
            encoded_layers, pooled_output = self.bert(ids, segments, attention_mask=mask)
            encoder1 = encoded_layers[-1][:, 0, :]
            encoder2 = pooled_output
            encoder = torch.cat([encoder1, encoder2], -1)
            encoder = F.dropout(encoder, p=0.6, training=self.training)
            score = self.linear(encoder)
            scores.append(score)
        return F.log_softmax(torch.cat(scores, 1), dim=-1)

